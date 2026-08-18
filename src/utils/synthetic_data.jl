# Build a complete synthetic MR-STAT experiment (phantom, geometry, coils,
# B₁ map, noiseless raw data) for a given forward model, so the example
# reconstructions and the validation scripts have a ready-made problem.
#
# The geometry must agree with the forward model: an N×N image acquired
# over K phase-encode cycles uses K*N readouts, and the model produces
# exactly nechos(model) echos per voxel — for a BlochModel that is the
# length of its RF train, for the surrogate the 600 readouts baked into its
# checkpoint. The function refuses mismatched geometry instead of silently
# producing an inconsistent experiment.
#
# The raw data is generated with the given model itself, so data and
# reconstruction share one forward model and the recovered maps should
# match the phantom essentially exactly (a self-consistency test of the
# pipeline). Cross-model experiments — reconstructing one model's data
# with the other — are built in scripts/ from the same pieces.
function generate_synthetic_data(model::ForwardModel; N=120, K=5)
    # total number of TRs — must match the echo train the forward model produces
    nTR = K * N
    nTR == nechos(model) || error(
        "geometry mismatch: N=$N, K=$K gives $(K*N) readouts but the " *
        "forward model produces $(nechos(model)) echos per voxel; choose N*K == nechos(model)")

    # Make coordinates
    FOVˣ = 22.4 # cm
    FOVʸ = 22.4 # cm
    Δx = FOVˣ / N # cm
    Δy = FOVˣ / N # cm
    x = -FOVˣ/2:Δx:FOVˣ/2-Δx |> collect # cm
    y = -FOVʸ/2:Δy:FOVʸ/2-Δy |> collect # cm
    z = [0.0]
    coordinates = @coordinates x y z # Takes Iterators.product and convert to a StructArray of BlochSimulators.Coordinates

    # Make trajectory
    Δt = 5e-6
    py_min = -N ÷ 2
    py_max = N ÷ 2 - 1
    py = repeat(py_min:py_max, K)

    Δkˣ = 2π / FOVˣ
    Δkʸ = 2π / FOVʸ
    k_start_readout = [(-N / 2 * Δkˣ) + im * (py[r] * Δkʸ) for r in 1:nTR]
    Δk_adc = Δkˣ

    trajectory = CartesianTrajectory2D(nTR, N, Δt, k_start_readout, Δk_adc, py, 1)

    # Make phantom (T₁, T₂, ρˣ, ρʸ per voxel)
    phantom = make_phantom(N)

    # Transmit field: the per-voxel B₁ map. It is a known, fixed input to
    # the forward model (the reconstruction estimates T₁, T₂ and ρ, not B₁).
    # A smooth radial variation stands in for realistic transmit
    # inhomogeneity; its range [0.9, 1.1] stays safely inside [0.8, 1.2],
    # the range the surrogate was trained on, so the same experiment works
    # for both forward models.
    r² = [ (xᵢ^2 + yⱼ^2) / ((FOVˣ/2)^2 + (FOVʸ/2)^2) for xᵢ in x, yⱼ in y ]
    transmit_field = @. 1.1 - 0.2 * r²   # ∈ [0.9, 1.1]

    # Make coil sensitivities
    coil₁ = complex.(repeat(LinRange(0.8, 1.2, N), 1, N))
    if NUM_COILS == 1
        coil_sensitivities = reshape(coil₁, N^2, 1)
    elseif NUM_COILS == 2
        coil₂ = coil₁'
        coil_sensitivities = [vec(coil₁) ;; vec(coil₂)]
    else
        @error "NUM_COILS is not 1 or 2"
    end

    # Set precision and send to gpu
    phantom = gpu(f32(vec(phantom)))
    trajectory = gpu(f32(trajectory))
    coil_sensitivities = gpu(f32(coil_sensitivities))
    coordinates = gpu(f32(vec(coordinates)))
    transmit_field = f32(vec(transmit_field))

    # Assemble the complete per-voxel tissue parameters the forward model
    # needs: the phantom's T₁/T₂/ρ plus the B₁ map. Constructed from the
    # component arrays directly, since everything is already on the GPU.
    parameters = StructArray{T₁T₂B₁ρˣρʸ{Float32}}((
        phantom.T₁, phantom.T₂, gpu(transmit_field), phantom.ρˣ, phantom.ρʸ))

    # Generate the noiseless raw data: echo trains from the forward model,
    # then the same phase-encoding and volume-integration operators the
    # objective uses. (Those operators are signal-model linear algebra
    # shared by every forward model.)
    resource = CUDALibs()
    echos = compute_echos(model, parameters)
    phase_encoding!(echos, trajectory, coordinates)
    raw_data = magnetization_to_signal(resource, echos, parameters, trajectory, coordinates, coil_sensitivities)

    # The first five returns are exactly mrstat_recon's data arguments, in
    # order (after the model itself). The phantom (ground-truth maps) is
    # appended last so callers can compute reconstruction errors or
    # generate alternative raw data from the same truth.
    return raw_data, coordinates, coil_sensitivities, trajectory, transmit_field, phantom
end
