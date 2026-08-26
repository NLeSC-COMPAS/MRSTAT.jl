# Standalone smoke test for the shim: no CUDA.jl, no MRSTAT — just Libdl.
# Loads libtrtjl.so, opens the engine, prints its I/O contract, closes.
#
#   julia deps/tensorrt/smoke.jl [path/to/model.engine]
#
# (With no CUDA context current, TensorRT's own CUDA runtime binds device 0's
# primary context — fine for a metadata dump.)
using Libdl

shim = joinpath(@__DIR__, "libtrtjl.so")
isfile(shim) || error("shim not built — run deps/tensorrt/build_shim.sh first")
engine = isempty(ARGS) ?
    normpath(joinpath(@__DIR__, "..", "..", "..", "NeuralBloch", "outputs", "model.engine")) :
    ARGS[1]
isfile(engine) || error("engine not found at $engine")

lib = dlopen(shim)
sym(s) = dlsym(lib, s)
ver(s) = ccall(sym(s), Int32, ())
println("shim built against TensorRT ", ver(:trtjl_build_version),
        ", runtime reports ", ver(:trtjl_lib_version))

h = ccall(sym(:trtjl_open), Ptr{Cvoid}, (Cstring,), engine)
h == C_NULL && error("open failed: ",
    unsafe_string(ccall(sym(:trtjl_last_error), Cstring, (Ptr{Cvoid},), C_NULL)))

n = ccall(sym(:trtjl_nb_io), Int32, (Ptr{Cvoid},), h)
println("engine ", engine, ": ", n, " I/O tensors")
dims = zeros(Int64, 8)
for i in 0:n-1
    name = unsafe_string(ccall(sym(:trtjl_io_name), Cstring, (Ptr{Cvoid}, Int32), h, i))
    isin = ccall(sym(:trtjl_is_input), Int32, (Ptr{Cvoid}, Cstring), h, name)
    es = ccall(sym(:trtjl_elt_size), Int32, (Ptr{Cvoid}, Cstring), h, name)
    nd = ccall(sym(:trtjl_engine_shape), Int32, (Ptr{Cvoid}, Cstring, Ptr{Int64}), h, name, dims)
    println("  ", isin == 1 ? "input " : "output", "  ", rpad(name, 8),
            " elt=", es, "B  dims=", Tuple(dims[1:nd]), "   (TensorRT order, -1 dynamic)")
    if isin == 1
        for (sel, lab) in ((0, "MIN"), (1, "OPT"), (2, "MAX"))
            np = ccall(sym(:trtjl_profile_shape), Int32,
                       (Ptr{Cvoid}, Cstring, Int32, Ptr{Int64}), h, name, sel, dims)
            println("      profile ", lab, " = ", Tuple(dims[1:np]))
        end
    end
end
println("context scratch: ",
    round(ccall(sym(:trtjl_device_mem_size), Int64, (Ptr{Cvoid},), h) / 2^20; digits=1),
    " MiB, aux streams: ", ccall(sym(:trtjl_nb_aux_streams), Int32, (Ptr{Cvoid},), h))
ccall(sym(:trtjl_close), Cvoid, (Ptr{Cvoid},), h)
println("smoke ok")
