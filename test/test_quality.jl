# Package-quality checks via Aqua.jl: method ambiguities, undefined exports,
# unbound type parameters, stale dependencies, missing compat bounds, type
# piracy. These run without a GPU.

using Aqua

@testset "package quality (Aqua)" begin
    Aqua.test_all(MRSTAT)
end
