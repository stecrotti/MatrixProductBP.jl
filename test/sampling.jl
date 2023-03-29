T = 2

A = [0 1 1 1; 1 0 0 0; 1 0 0 0; 1 0 0 0]
g = IndexedGraph(A)
N = size(A, 1)

λ = 0.5
ρ = 0.4
γ = 0.5
ρ = 0.2

sis = SIS(g, λ, ρ, T; γ)
bp = mpbp(sis)
rng = MersenneTwister(111)

draw_node_observations!(bp, N, last_time=true; rng)

sms = sample(bp, 10; showprogress=false)
m = marginals(sms)
pb = pair_marginals(sms)

f(x,i) = x-1
c = autocovariances(f, sms)

av, va = continuous_sis_sampler(sis, T, λ, ρ; nsamples = 10^4, sites=1,
    discard_dead_epidemics=true)

# just check that it runs without errors
@testset "sampling" begin
    @test true
end