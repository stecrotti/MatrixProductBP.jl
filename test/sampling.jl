T = 2

A = [0 1 1 1; 1 0 0 0; 1 0 0 0; 1 0 0 0]
g = IndexedGraph(A)
N = size(A, 1)

λ = 0.5
ρ = 0.4
γ = 0.5
ρ = 0.2
σ = 0.1

sirs = SIRS(g, λ, ρ, σ, T; γ)
bp = mpbp(sirs)
rng = MersenneTwister(111)

draw_node_observations!(bp, N, last_time=true; rng)

sms = sample(bp, 10; showprogress=false)
m = marginals(sms)

f(x,i) = x-1
c = autocovariances(f, sms)

# just check that it runs without errors
@testset "sampling" begin
    @test true
end