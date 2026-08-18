# Unit tests for generate_synthetic_data with both forward models:
# geometry, shapes, ranges, and the echo-count consistency guard.
# Uses the shared SURROGATE fixture.

using Test
using CUDA
using StructArrays: StructArray
using BlochSimulators
using BlochSimulators: nreadouts, nsamples

# The contracts every generated experiment must satisfy, whatever the model.
function synthetic_data_tests(model, label; N = 120, K = 5)
    @testset "generate_synthetic_data: $label" begin
        raw, coordinates, coils, trajectory, transmit_field, phantom =
            generate_synthetic_data(model; N, K)

        @test nreadouts(trajectory) == nechos(model) == N * K
        @test length(coordinates) == N^2

        # Raw data: BlochSimulators' Cartesian fast path returns
        # (samples per readout, readouts, coils); the objective consumes
        # the same layout on both sides of the residual.
        @test size(raw) == (trajectory.nsamplesperreadout, nreadouts(trajectory), MRSTAT.NUM_COILS)
        @test length(raw) == nsamples(trajectory) * MRSTAT.NUM_COILS
        @test eltype(raw) <: Complex
        @test all(isfinite, Array(raw))

        # Transmit field: per-voxel, inside the surrogate's trained B₁ range
        # (the shared experiment design keeps it valid for both models).
        @test length(transmit_field) == N^2
        @test all(b -> 0.8 ≤ b ≤ 1.2, transmit_field)

        # Phantom: ground-truth maps as a GPU StructArray, values inside
        # the surrogate's trained box (so either model can reconstruct the
        # same experiment without extrapolating).
        @test phantom isa StructArray
        @test phantom.T₁ isa CuVector{Float32}
        @test length(phantom) == N^2
        @test all(0.1f0 .≤ Array(phantom.T₁) .≤ 5.0f0)
        @test all(0.01f0 .≤ Array(phantom.T₂) .≤ 2.0f0)
    end
    return nothing
end

# Surrogate: geometry is fixed by the checkpoint's 600 readouts.
synthetic_data_tests(SURROGATE, "SurrogateModel"; N = 120, K = 5)

# Bloch: any geometry works as long as the sequence length matches; use the
# original example's ramp sequence at the same shared geometry.
let N = 120, K = 5, nTR = N * K
    RF = complex.(collect(range(1, 90; length = nTR)))
    bloch = BlochModel(gpu(f32(FISP2D(RF, ones(nTR, 1) .|> complex, 0.010, 0.006, 32, 0.025))))
    synthetic_data_tests(bloch, "BlochModel"; N, K)

    # Geometry that cannot produce the model's echo count is refused
    # instead of silently mis-shaping the experiment.
    @test_throws ErrorException generate_synthetic_data(bloch; N = 100, K = 5)
    @test_throws ErrorException generate_synthetic_data(SURROGATE; N = 100, K = 5)
end
