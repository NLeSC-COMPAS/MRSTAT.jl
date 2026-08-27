# Parity validation: the Julia surrogate wrapper vs pure-python execution.
#
# Run with:
#   source env.sh                          # binds PythonCall to the torch venv
#   julia --project=. scripts/validate_python_parity.jl
#
# The reference path is the NeuralBloch repository's own way of running the
# model — a fresh FCModel built from their src/model.py, torch.load of the
# checkpoint, plain torch tensors of physical [T₁, T₂, B₁] as input, and a
# vanilla forward under no_grad. None of the wrapper's machinery (zero-copy
# crossings, complex views, input staging) touches the reference. The
# comparison therefore isolates exactly what the Julia wrapper adds — and
# shows it adds nothing numerically.
#
# Two comparison layers:
#   1. Execution parity — python receives the *identical float32 rows* the
#      wrapper staged, so input rounding is out of the picture. Both paths
#      run the same model code and kernels on the same device (the log
#      transform included — it lives inside FCModel.forward), so the
#      outputs are expected to match EXACTLY (max |Δ| = 0).
#   2. End-to-end parity — python casts the original Float64 physical
#      parameters to float32 itself, as NeuralBloch's own pipeline feeds
#      the model, while the wrapper's inputs went through the
#      reconstruction's f32 conversion. IEEE double→single rounding is
#      deterministic and identical on both routes, so this too is expected
#      exact; the tolerance is a defensive gate, not an expectation.

using MRSTAT
using MRSTAT.ForwardModels
using MRSTAT.ForwardModels: _stage_input!, _dim, _num_layers
using NeuralBloch   # activates the surrogate's package extension
using BlochSimulators: T₁T₂B₁ρˣρʸ, gpu, f32
using StructArrays: StructArray
using CUDA
using Random
using PythonCall: pyimport, pyconvert, pywith, pydict, pylist

const NTOL_E2E = 1e-4   # end-to-end relative gate (expected exact; defensive)

surrogate = load_surrogate()
NTR = nechos(surrogate)
neuralbloch_dir = neuralbloch_repo()
# The same artifact the wrapper loaded, through the second code path — the
# comparison is between two ways of running one checkpoint, so it must not
# read a (possibly stale) sibling copy.
model_pt = joinpath(models_dir(), "model.pt")

println("═"^70)
println("Pure-python reference model (their code path)")
println("═"^70)

tm = pyimport("torch")
sys = pyimport("sys")
pyconvert(Bool, sys.path.__contains__(neuralbloch_dir)) || sys.path.insert(0, neuralbloch_dir)
FCModel = pyimport("src.model").FCModel

ckpt = tm.load(model_pt; map_location = "cpu")
sd = ckpt["model_state"]
keys = pyconvert(Vector{String}, pylist(sd.keys()))
config = pydict(Dict{String,Int}(
    "input_dim"          => _dim(sd, "subnet_signal.net.0.weight", 2),
    "NB_signal"          => _dim(sd, "U_signal.0.weight", 2),
    "NB_dT1"             => _dim(sd, "U_dT1.0.weight", 2),
    "NB_dT2"             => _dim(sd, "U_dT2.0.weight", 2),
    "hidden_size_signal" => _dim(sd, "subnet_signal.net.0.weight", 1),
    "hidden_size_dT1"    => _dim(sd, "subnet_dT1.net.0.weight", 1),
    "hidden_size_dT2"    => _dim(sd, "subnet_dT2.net.0.weight", 1),
    "num_layers_signal"  => _num_layers(keys, "subnet_signal"),
    "num_layers_dT1"     => _num_layers(keys, "subnet_dT1"),
    "num_layers_dT2"     => _num_layers(keys, "subnet_dT2"),
))
ref_model = FCModel(pyconvert(Int, ckpt["NTR"]), config)
ref_model.load_state_dict(sd)          # strict — the training layout, verbatim
ref_model = ref_model.cuda()
ref_model.eval()
println("  reference FCModel loaded (cuda, eval)")

# Run the reference on a (N, 3) float32 input given as plain nested Python
# lists (Float32 → Python float is exact) and return the three outputs as
# Julia (N, NTR, 2) arrays — the model's native paired layout.
function reference_forward(rows::Vector{Vector{Float64}})
    x = tm.tensor(pylist(pylist.(rows)); dtype = tm.float32, device = "cuda")
    out = pywith(tm.no_grad()) do _
        ref_model(x)
    end
    toarr(t) = pyconvert(Array{Float32,3}, t.detach().cpu().numpy())
    return toarr(out[0]), toarr(out[1]), toarr(out[2])
end

# Assemble the wrapper-convention complex matrix from a paired (N, NTR, 2)
# reference output: echos_ref[r, v] = a[v, r, 1] + i·a[v, r, 2].
tocomplex(a) = permutedims(ComplexF32.(a[:, :, 1] .+ im .* a[:, :, 2]))

maxabs(a, b) = maximum(abs, a .- b)
relerr(a, b) = maximum(abs.(a .- b) ./ (abs.(b) .+ eps(Float32)))

# ---------------------------------------------------------------------------
println("═"^70)
println("Layer 1: execution parity (identical float32 inputs)")
println("═"^70)

rng = Xoshiro(20260818)
n = 64
T₁ = exp.(rand(rng, n) .* (log(4.5) - log(0.15)) .+ log(0.15))
T₂ = min.(exp.(rand(rng, n) .* (log(1.5) - log(0.015)) .+ log(0.015)), 0.9 .* T₁)
B₁ = 0.82 .+ 0.36 .* rand(rng, n)
params = gpu(f32(StructArray(map(T₁T₂B₁ρˣρʸ, T₁, T₂, B₁, ones(n), zeros(n)))))

echos, ∂echos = compute_echos_and_derivatives(surrogate, params)

# The staged buffer holds exactly the float32 rows the wrapper fed the
# model; hand those same numbers to the pure-python path.
Xcpu = Array(surrogate.Xbuf)                       # (3, n) Float32
rows = [Float64.(Xcpu[:, j]) for j in 1:n]         # exact Float32 → Float64
m_ref, d1_ref, d2_ref = reference_forward(rows)

Δ_sig = maxabs(Array(echos), tocomplex(m_ref))
Δ_dt1 = maxabs(Array(∂echos.∂T₁), tocomplex(d1_ref))
Δ_dt2 = maxabs(Array(∂echos.∂T₂), tocomplex(d2_ref))
println("  max |julia − python| over $(n)×$(NTR) values:")
println("    signal : $Δ_sig")
println("    ∂/∂T₁  : $Δ_dt1")
println("    ∂/∂T₂  : $Δ_dt2")
@assert Δ_sig == 0 && Δ_dt1 == 0 && Δ_dt2 == 0 """
    wrapper output differs from pure-python execution on identical inputs —
    the zero-copy path is not numerically transparent."""
println("  ✓ EXACT: zero-copy crossings and complex views change nothing")

# ---------------------------------------------------------------------------
println("═"^70)
println("Layer 2: end-to-end parity (python casts its own inputs)")
println("═"^70)

# The NeuralBloch pipeline feeds the model physical parameters cast to
# float32; the wrapper's inputs went through the reconstruction's f32
# conversion instead. IEEE double→single rounding is deterministic and
# identical on both routes, so this is expected exact too — the tolerance
# only guards against an input-staging convention drifting.
rows_e2e = [[Float64(T₁[j]), Float64(T₂[j]), Float64(B₁[j])] for j in 1:n]
m_e2e, d1_e2e, d2_e2e = reference_forward(rows_e2e)

r_sig = relerr(Array(echos), tocomplex(m_e2e))
r_dt1 = relerr(Array(∂echos.∂T₁), tocomplex(d1_e2e))
r_dt2 = relerr(Array(∂echos.∂T₂), tocomplex(d2_e2e))
println("  max relative difference (both sides cast the same physical values):")
println("    signal : $r_sig")
println("    ∂/∂T₁  : $r_dt1")
println("    ∂/∂T₂  : $r_dt2")
@assert r_sig < NTOL_E2E && r_dt1 < NTOL_E2E && r_dt2 < NTOL_E2E """
    end-to-end difference exceeds the gate — the input staging (units,
    row order, rounding route) disagrees with the python pipeline."""
println("  ✓ end-to-end within $(NTOL_E2E) of the python data pipeline")

println()
println("python parity validation PASSED")
