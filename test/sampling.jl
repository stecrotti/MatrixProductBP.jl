@testset "Sampling" begin

    T = 2

    A = [0 1 1 1; 1 0 0 0; 1 0 0 0; 1 0 0 0]
    g = IndexedGraph(A)
    N = size(A, 1)

    λ = 0.5
    ρ = 0.4
    γ = 0.5
    ρ = 0.2
    α = 0.1

    sis = SIS(g, λ, ρ, T; γ, α)
    bp = mpbp(sis)
    rng = MersenneTwister(111)

    @test is_free_dynamics(bp)

    draw_node_observations!(bp, N, last_time=true; rng, softinf=1e2)

    sms = sample(bp, 10; showprogress=false, rng)
    m = marginals(sms)
    pb = pair_marginals(sms)

    f(x,i) = x-1
    c = autocovariances(f, sms)

    @testset "sampling - SoftMargin" begin
        @test all(all(all(0 ≤ x.val ≤ 1 for x in mit) for mit in mi) for mi in m)
        @test all(all(all(0 ≤ x.val ≤ 1 for x in pbit) for pbit in pbi) for pbi in pb)
        @test all(all(all(-1 ≤ x.val ≤ 1 for x in cit) for cit in ci) for ci in c)
    end

    @testset "sampling - Gillespie - reproducibility" begin
        av, va = continuous_sis_sampler(sis, T, λ, ρ, α; nsamples = 10^4, sites=1,
        discard_dead_epidemics=true, rng = MersenneTwister(0))
        av2, va2 = continuous_sis_sampler(sis, T, λ, ρ, α; nsamples = 10^4, sites=1,
        discard_dead_epidemics=true, rng = MersenneTwister(0))
        @test av2 == av && va2 == va  
    end
end