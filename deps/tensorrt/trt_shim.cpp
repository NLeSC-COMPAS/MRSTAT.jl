// ============================================================================
// trt_shim.cpp — minimal C ABI over the TensorRT C++ runtime, for Julia ccall.
//
// Inference-only: deserialize a prebuilt engine, bind device buffers, enqueue
// on a caller-provided CUDA stream, and optionally capture/replay a CUDA
// graph. Engine *building* is deliberately absent — engines come prebuilt
// from the NeuralBloch repository.
//
// ABI rules:
//   - POD types only (void*, int32_t, int64_t, const char*);
//   - C++ exceptions never cross the boundary: every entry point is noexcept
//     and catches; errors are returned as codes, with a human-readable
//     message retrievable via trtjl_last_error(handle) — pass NULL for
//     failures of trtjl_open itself;
//   - the logger only writes to a buffer / stderr; it must never call back
//     into Julia (it can fire on non-Julia threads).
//
// CUDA-context assumptions: TensorRT (statically linked cudart) uses the
// driver context current on the calling thread. The Julia side makes CUDA.jl
// bind the device's primary context before calling trtjl_open, so no
// cudaSetDevice happens here — one context everywhere is what makes raw
// CuArray device pointers directly bindable (zero-copy I/O).
//
// CUDA-graph capture happens on a PRIVATE non-blocking stream owned by this
// handle, never on the caller's stream: while the capturing ccall is in
// flight, Julia's GC (on other threads) may issue stream-ordered frees on
// task streams, and anything recorded on a capturing stream would become part
// of the graph. Replay then launches the (stream-agnostic) graph on whatever
// stream the caller passes.
// ============================================================================

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

using namespace nvinfer1;

namespace {

// Last message the TensorRT logger classified as an error — appended to the
// shim's own error strings so Julia sees TensorRT's diagnosis, not just ours.
char g_last_log[1024] = {0};

class ShimLogger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kERROR) {
            std::snprintf(g_last_log, sizeof(g_last_log), "%s", msg);
        }
        if (severity <= Severity::kWARNING) {
            std::fprintf(stderr, "[TensorRT] %s\n", msg);
        }
    }
};

ShimLogger g_logger;          // must outlive every runtime created with it
char g_err[2048] = {0};       // error slot for trtjl_open (no handle yet)

struct Ctx {
    IRuntime* runtime = nullptr;
    ICudaEngine* engine = nullptr;
    IExecutionContext* context = nullptr;
    cudaStream_t capture_stream = nullptr;   // private, non-blocking
    cudaGraphExec_t graph_exec = nullptr;
    char err[2048] = {0};
};

void set_err(char* slot, const char* fmt, ...) noexcept {
    va_list args;
    va_start(args, fmt);
    const int n = std::vsnprintf(slot, 2048, fmt, args);
    va_end(args);
    if (g_last_log[0] != '\0' && n > 0 && n < 2048) {
        std::snprintf(slot + n, 2048 - n, " | TensorRT: %s", g_last_log);
    }
    g_last_log[0] = '\0';
}

Ctx* ctx(void* h) noexcept { return static_cast<Ctx*>(h); }

void destroy(Ctx* c) noexcept {
    if (!c) return;
    // Ignore CUDA errors here: at process exit the driver may already be
    // tearing down, and finalizer-triggered closes must never crash.
    if (c->graph_exec) cudaGraphExecDestroy(c->graph_exec);
    if (c->capture_stream) cudaStreamDestroy(c->capture_stream);
    delete c->context;   // TensorRT 10+: destruction is plain delete
    delete c->engine;
    delete c->runtime;
    delete c;
}

} // namespace

extern "C" {

// ------------------------------------------------------------------ versions
int32_t trtjl_build_version(void) noexcept { return NV_TENSORRT_VERSION; }
int32_t trtjl_lib_version(void) noexcept { return getInferLibVersion(); }

const char* trtjl_last_error(void* h) noexcept {
    return h ? ctx(h)->err : g_err;
}

// ----------------------------------------------------------------- lifecycle
void* trtjl_open(const char* engine_path) noexcept {
    try {
        g_last_log[0] = '\0';
        std::ifstream f(engine_path, std::ios::binary | std::ios::ate);
        if (!f) {
            set_err(g_err, "cannot read engine file %s", engine_path);
            return nullptr;
        }
        const std::streamsize n = f.tellg();
        f.seekg(0, std::ios::beg);
        std::vector<char> blob(static_cast<size_t>(n));
        if (!f.read(blob.data(), n)) {
            set_err(g_err, "short read on engine file %s", engine_path);
            return nullptr;
        }

        Ctx* c = new Ctx();
        c->runtime = createInferRuntime(g_logger);
        if (!c->runtime) {
            set_err(g_err, "createInferRuntime failed");
            destroy(c);
            return nullptr;
        }
        c->engine = c->runtime->deserializeCudaEngine(blob.data(), blob.size());
        if (!c->engine) {
            set_err(g_err, "deserializeCudaEngine failed for %s (TensorRT "
                    "version / GPU mismatch with the engine build?)", engine_path);
            destroy(c);
            return nullptr;
        }
        // Default allocation strategy (kSTATIC): the context's scratch memory
        // (getDeviceMemorySizeV2, sized for the profile MAX) is allocated
        // eagerly here — the one big allocation of the whole setup.
        c->context = c->engine->createExecutionContext();
        if (!c->context) {
            set_err(g_err, "createExecutionContext failed (out of GPU memory?)");
            destroy(c);
            return nullptr;
        }
        if (cudaStreamCreateWithFlags(&c->capture_stream, cudaStreamNonBlocking)
            != cudaSuccess) {
            set_err(g_err, "cudaStreamCreateWithFlags failed");
            destroy(c);
            return nullptr;
        }
        return c;
    } catch (...) {
        set_err(g_err, "unexpected C++ exception in trtjl_open");
        return nullptr;
    }
}

void trtjl_close(void* h) noexcept {
    destroy(ctx(h));
}

// ------------------------------------------------- metadata (load time only)
int32_t trtjl_nb_io(void* h) noexcept {
    return ctx(h)->engine->getNbIOTensors();
}

const char* trtjl_io_name(void* h, int32_t i) noexcept {
    return ctx(h)->engine->getIOTensorName(i);
}

int32_t trtjl_is_input(void* h, const char* name) noexcept {
    return ctx(h)->engine->getTensorIOMode(name) == TensorIOMode::kINPUT ? 1 : 0;
}

// Element size in bytes, straight from the engine's dtype — the Julia side
// asserts sizeof(its element type) against this instead of hardcoding
// TensorRT's DataType enum values. -1 = sub-byte / unsupported for I/O.
int32_t trtjl_elt_size(void* h, const char* name) noexcept {
    switch (ctx(h)->engine->getTensorDataType(name)) {
        case DataType::kFLOAT: return 4;
        case DataType::kHALF:  return 2;
        case DataType::kBF16:  return 2;
        case DataType::kINT8:  return 1;
        case DataType::kUINT8: return 1;
        case DataType::kBOOL:  return 1;
        case DataType::kFP8:   return 1;
        case DataType::kINT32: return 4;
        case DataType::kINT64: return 8;
        default:               return -1;
    }
}

// Engine-level (static) shape: dims in TensorRT (row-major) order, -1 marks a
// dynamic dimension. Returns nbDims, or -1 on failure. dims must hold 8.
int32_t trtjl_engine_shape(void* h, const char* name, int64_t* dims) noexcept {
    const Dims d = ctx(h)->engine->getTensorShape(name);
    if (d.nbDims < 0) return -1;
    for (int32_t i = 0; i < d.nbDims; ++i) dims[i] = d.d[i];
    return d.nbDims;
}

// Optimization-profile shape of an input tensor (profile 0).
// sel: 0 = MIN, 1 = OPT, 2 = MAX. Returns nbDims, or -1 on failure.
int32_t trtjl_profile_shape(void* h, const char* name, int32_t sel,
                            int64_t* dims) noexcept {
    if (sel < 0 || sel > 2) return -1;
    const Dims d = ctx(h)->engine->getProfileShape(
        name, 0, static_cast<OptProfileSelector>(sel));
    if (d.nbDims < 0) return -1;
    for (int32_t i = 0; i < d.nbDims; ++i) dims[i] = d.d[i];
    return d.nbDims;
}

int64_t trtjl_device_mem_size(void* h) noexcept {
    return ctx(h)->engine->getDeviceMemorySizeV2();
}

int32_t trtjl_nb_aux_streams(void* h) noexcept {
    return ctx(h)->engine->getNbAuxStreams();
}

// ------------------------------------------- shapes & addresses (per rebind)
int32_t trtjl_set_input_shape(void* h, const char* name, const int64_t* dims,
                              int32_t nb) noexcept {
    Ctx* c = ctx(h);
    Dims d;
    d.nbDims = nb;
    for (int32_t i = 0; i < nb; ++i) d.d[i] = dims[i];
    if (!c->context->setInputShape(name, d)) {
        set_err(c->err, "setInputShape(%s) rejected (outside the engine's "
                "optimization profile?)", name);
        return -1;
    }
    return 0;
}

// Context-level (resolved) shape — valid for outputs once all input shapes
// are set. Returns nbDims, or -1 on failure.
int32_t trtjl_current_shape(void* h, const char* name, int64_t* dims) noexcept {
    const Dims d = ctx(h)->context->getTensorShape(name);
    if (d.nbDims < 0) return -1;
    for (int32_t i = 0; i < d.nbDims; ++i) dims[i] = d.d[i];
    return d.nbDims;
}

int32_t trtjl_set_address(void* h, const char* name, void* dptr) noexcept {
    Ctx* c = ctx(h);
    if (!c->context->setTensorAddress(name, dptr)) {
        set_err(c->err, "setTensorAddress(%s) rejected (misaligned or null "
                "device pointer?)", name);
        return -1;
    }
    return 0;
}

// ------------------------------------------------------------------ hot path
int32_t trtjl_enqueue(void* h, void* stream) noexcept {
    Ctx* c = ctx(h);
    if (!c->context->enqueueV3(static_cast<cudaStream_t>(stream))) {
        set_err(c->err, "enqueueV3 failed (are all I/O tensor addresses and "
                "the input shape set?)");
        return -1;
    }
    return 0;
}

// Capture the current shape/address binding as a CUDA graph. Preconditions:
// input shape and all tensor addresses set. Any previously captured graph is
// destroyed FIRST, so a failed re-capture can never leave a stale executable
// graph holding old buffer addresses.
int32_t trtjl_capture(void* h, void* warm_stream) noexcept {
    Ctx* c = ctx(h);
    if (c->graph_exec) {
        cudaGraphExecDestroy(c->graph_exec);
        c->graph_exec = nullptr;
    }
    cudaStream_t ws = static_cast<cudaStream_t>(warm_stream);

    // Warm-up run: flushes lazy kernel selection / internal allocations that
    // must not be recorded into the graph. Synchronized so nothing of it can
    // leak into the capture below.
    if (!c->context->enqueueV3(ws)) {
        set_err(c->err, "capture warm-up enqueueV3 failed");
        return -1;
    }
    if (cudaStreamSynchronize(ws) != cudaSuccess) {
        set_err(c->err, "capture warm-up cudaStreamSynchronize failed");
        return -2;
    }

    if (cudaStreamBeginCapture(c->capture_stream,
                               cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
        set_err(c->err, "cudaStreamBeginCapture failed");
        return -3;
    }
    const bool enq_ok = c->context->enqueueV3(c->capture_stream);
    cudaGraph_t graph = nullptr;
    const cudaError_t end_rc = cudaStreamEndCapture(c->capture_stream, &graph);
    if (!enq_ok || end_rc != cudaSuccess || graph == nullptr) {
        if (graph) cudaGraphDestroy(graph);
        set_err(c->err, "graph capture failed (enqueueV3 %s, EndCapture %s)",
                enq_ok ? "ok" : "failed",
                end_rc == cudaSuccess ? "ok" : cudaGetErrorName(end_rc));
        return -4;
    }
    const cudaError_t inst_rc = cudaGraphInstantiate(&c->graph_exec, graph, 0);
    cudaGraphDestroy(graph);
    if (inst_rc != cudaSuccess) {
        c->graph_exec = nullptr;
        set_err(c->err, "cudaGraphInstantiate failed: %s",
                cudaGetErrorName(inst_rc));
        return -5;
    }
    return 0;
}

int32_t trtjl_has_graph(void* h) noexcept {
    return ctx(h)->graph_exec ? 1 : 0;
}

int32_t trtjl_replay(void* h, void* stream) noexcept {
    Ctx* c = ctx(h);
    if (!c->graph_exec) {
        set_err(c->err, "trtjl_replay called with no captured graph");
        return -1;
    }
    const cudaError_t rc =
        cudaGraphLaunch(c->graph_exec, static_cast<cudaStream_t>(stream));
    if (rc != cudaSuccess) {
        set_err(c->err, "cudaGraphLaunch failed: %s", cudaGetErrorName(rc));
        return -2;
    }
    return 0;
}

} // extern "C"
