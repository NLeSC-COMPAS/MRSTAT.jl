function objective(optimpars::Vector{<:Real}, resource, mode, raw_data, model::ForwardModel, coordinates, coil_sensitivities, trajectory, transmit_field)

    # We compute the residual rᵢ = ||d Σᵢ (dᵢ - M(T₁,T₂,B₁,B₀)*Cᵢ*ρ)
    # f = (1/2) * |r|^2
    # The gradient is computed as g = ℜ(Jᴴr)

    # mode 0 -> compute f and r only
    # mode 1 -> compute f, r and g
    # mode 2 -> compute f, r, g and assemble approximate Hessian

    # Convert optimpars (Vector{<:Real}) to Vector{<:AbstractTissueParameters} to be used by the forward model
    parameters = optim_to_physical_pars(optimpars, transmit_field)

    # Convert to single precision and send to gpu device
    parameters = gpu(f32(parameters))

    # The forward model is pluggable (see src/ForwardModels/): a Bloch/EPG
    # simulation or the neural surrogate, both behind the same two-function
    # interface. Mode 0 needs only the cost and residual, so it asks for
    # echos alone — for the surrogate that skips the derivative
    # sub-networks entirely, and for the Bloch model it skips two extra
    # finite-difference simulations.
    if mode == 0

        # Compute magnetization at echo times
        echos = compute_echos(model, parameters)

        # Apply phase encoding
        phase_encoding!(echos, trajectory, coordinates)

        # Compute signal
        s = magnetization_to_signal(resource, echos, parameters, trajectory, coordinates, coil_sensitivities)

        # Compute residual r
        r = s - raw_data;

        # From array to vector of svectors
        r = reshape(collect(r), :, NUM_COILS)
        r = map(SVector{NUM_COILS}, eachrow(r)) |> gpu

        # Compute cost f
        f = 0.5 * sum(norm.(r).^2);

        return f, r

    elseif mode > 0

        # Compute the magnetization at echo times together with its partial
        # derivatives with respect to T₁ and T₂. The derivatives are in
        # physical units (per second) on purpose: the Jᴴv/Jv kernels below
        # convert to the log-parametrization themselves through their
        # lin_scale factors, so nothing must be pre-scaled here.
        echos, ∂echos = compute_echos_and_derivatives(model, parameters)

        # Apply phase encoding to the echos and to both derivative
        # components. Phase encoding multiplies every element by a scalar
        # phase, so encoding ∂echos component-by-component is mathematically
        # identical to encoding it as a whole — and it keeps each broadcast
        # on a plain CuMatrix{ComplexF32}, which the standard
        # phase_encoding! method handles directly.
        phase_encoding!(echos, trajectory, coordinates)
        phase_encoding!(∂echos.∂T₁, trajectory, coordinates)
        phase_encoding!(∂echos.∂T₂, trajectory, coordinates)

        # Compute signal
        s = magnetization_to_signal(resource, echos, parameters, trajectory, coordinates, coil_sensitivities)

        # Compute residual r
        r = s - raw_data;

        # From array to vector of svectors
        r = reshape(collect(r), :, NUM_COILS)
        r = map(SVector{NUM_COILS}, eachrow(r)) |> gpu

        # Compute cost f
        f = 0.5 * sum(norm.(r).^2);

        # Compute gradient

        coil_sensitivities = collect(coil_sensitivities)
        coil_sensitivities_svec = map(SVector{NUM_COILS}, eachrow(coil_sensitivities)) |> gpu

        g = Jᴴv(resource, echos, ∂echos, parameters, coil_sensitivities_svec, trajectory, coordinates, r)
        g = StructArray(g)
        g = reduce(vcat,fieldarrays(g))
        g = real.(g)
        g = collect(g)

        mode == 1 && return f, r, g

        # Make Gauss-Newton matrix multiply function
        reJᴴJ(x) = begin
            np = 4 # nr of reconstruction parameters per voxel
            x = reshape(x,:,np)
            x = map(SVector{4}, eachcol(x)...) |> gpu
            y = Jv(resource, echos, ∂echos, parameters, coil_sensitivities_svec, trajectory, coordinates, x)
            z = Jᴴv(resource, echos, ∂echos, parameters, coil_sensitivities_svec, trajectory, coordinates, y)
            return real.(reduce(vcat,fieldarrays(StructArray(z))))
        end

        H = LinearMap(
            v -> reJᴴJ(v),
            v -> v, # adjoint operation not used
        length(g),length(g));

        return f, r, g, H
    end
end

function optim_to_physical_pars(optimpars, transmit_field)

    optimpars = reshape(optimpars,:,4)
    T₁ = exp.(optimpars[:,1])
    T₂ = exp.(optimpars[:,2])
    B₁ = vec(transmit_field)
    ρˣ = optimpars[:,3]
    ρʸ = optimpars[:,4]

    return map(T₁T₂B₁ρˣρʸ, T₁, T₂, B₁, ρˣ, ρʸ) |> StructArray
end
