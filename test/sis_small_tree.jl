@testset "SIS small tree" begin
    T = 3

    A = [0 1 1 1; 1 0 0 0; 1 0 0 0; 1 0 0 0]
    g = IndexedGraph(A)
    N = size(A, 1)

    λ = 0.5
    ρ = 0.4
    γ = 0.5
    α = 0.1

    sis = SIS(g, λ, ρ, T; γ, α)
    bp = mpbp(sis)
    rng = MersenneTwister(111)
    X, _ = onesample(bp; rng)

    draw_node_observations!(bp.ϕ, X, N, last_time=true; rng)

    @testset "SIS small tree" begin
        svd_trunc = TruncBondMax(4)
        @show svd_trunc
        iterate!(bp, maxiter=10; svd_trunc, showprogress=false)

        b_bp = beliefs(bp)
        p_bp = [[bbb[2] for bbb in bb] for bb in b_bp]

        p_exact, Z_exact = exact_prob(bp)
        b_exact = exact_marginals(bp; p_exact)
        p_ex = [[bbb[2] for bbb in bb] for bb in b_exact]

        f_bethe = bethe_free_energy(bp)
        Z_bp = exp(-f_bethe)

        local f(x,i) = x-1

        r_bp = autocorrelations(f, bp)
        r_exact = exact_autocorrelations(f, bp; p_exact)

        c_bp = autocovariances(f, bp)
        c_exact = exact_autocovariances(f, bp; r = r_exact)

        pb_exact = exact_pair_marginals(bp)
        pb_bp = pair_beliefs(bp)[1]

        @test Z_exact ≈ Z_bp
        @test p_ex ≈ p_bp
        @test r_bp ≈ r_exact
        @test c_bp ≈ c_exact
        @test pb_bp ≈ pb_exact
    end

    @testset "RestrictedRecursiveBPFactor - RecursiveBPFactor generic methods" begin
        bpfake = MPBP(bp.g, [RestrictedRecursiveBPFactor.(w) for w in bp.w], bp.ϕ, bp.ψ, 
                        deepcopy(collect(bp.μ)), collect(bp.b), collect(bp.f))

        for i=1:20
            X, _ = onesample(bp; rng)
            @test logprob(bp, X) ≈ logprob(bpfake, X)
        end

        iterate!(bpfake, maxiter=10; svd_trunc, showprogress=false)

        @test beliefs(bpfake) ≈ beliefs(bp)

    end

    @testset "GenericFactor - extensive trace update test" begin
        bpfake = MPBP(bp.g, [GenericFactor.(w) for w in bp.w], bp.ϕ, bp.ψ, 
                        deepcopy(collect(bp.μ)), collect(bp.b), collect(bp.f))

        for i=1:20
            X, _ = onesample(bp; rng)
            @test logprob(bp, X) ≈ logprob(bpfake, X)
        end

        iterate!(bpfake, maxiter=10; svd_trunc, showprogress=false)

        @test beliefs(bpfake) ≈ beliefs(bp)
        pb_fake,_ = pair_beliefs(bpfake)
        pb, _ = pair_beliefs(bp)
        @test pb_fake ≈ pb
    end

    @testset "RecursiveTraceFactor" begin
        bpfake = MPBP(bp.g, [RecursiveTraceFactor.(w,2) for w in bp.w], bp.ϕ, bp.ψ, 
                        deepcopy(collect(bp.μ)), collect(bp.b), collect(bp.f))

        for i=1:20
            X, _ = onesample(bp; rng)
            @test logprob(bp, X) ≈ logprob(bpfake, X)
        end

        iterate!(bpfake, maxiter=10; svd_trunc=TruncBond(10), showprogress=false)

        @test beliefs(bpfake) ≈ beliefs(bp)

    end

    # observe everything and check that the free energy corresponds to the posterior of sample `X`
    reset!(bp; observations=true)
    draw_node_observations!(bp.ϕ, X, N*(T+1), last_time=false)
    svd_trunc = TruncBond(4)
    iterate!(bp, maxiter=50; svd_trunc, showprogress=false, tol=0)
    f_bethe = bethe_free_energy(bp)
    logl_bp = - f_bethe
    logp = logprob(bp, X)

    @testset "SIS small tree - observe everything" begin
        @test logl_bp ≈ logp
    end

    @testset "SIS small tree - periodic" begin
        bp = periodic_mpbp(sis)
        rng = MersenneTwister(111)
        X, _ = onesample(bp; rng)

        reset!(bp; observations=true)
        draw_node_observations!(bp.ϕ, X, N, last_time=true; rng)

        svd_trunc = TruncBondMax(10)
        iterate!(bp, maxiter=10; svd_trunc, showprogress=false)

        b_bp = beliefs(bp)
        p_bp = [[bbb[2] for bbb in bb] for bb in b_bp]

        p_exact, Z_exact = exact_prob(bp)
        b_exact = exact_marginals(bp; p_exact)
        p_ex = [[bbb[2] for bbb in bb] for bb in b_exact]

        f_bethe = bethe_free_energy(bp)
        Z_bp = exp(-f_bethe)

        local f(x,i) = x-1

        r_bp = autocorrelations(f, bp)
        r_exact = exact_autocorrelations(f, bp; p_exact)

        c_bp = autocovariances(f, bp)
        c_exact = exact_autocovariances(f, bp; r = r_exact)

        @test Z_exact ≈ Z_bp
        @test p_ex ≈ p_bp
        @test r_bp ≈ r_exact
        @test c_bp ≈ c_exact
    end

    @testset "SIS small tree - stationary" begin
        sis = SIS(g, λ, ρ, 0; γ, α)
        bp = mpbp_stationary(sis)
        svd_trunc = TruncVUMPS(10)
    
        iterate!(bp; tol=1e-14, maxiter=10, svd_trunc)
        local f(x,i) = x-1
        m_bp = [only(m) for m in means(f, bp)]

        bp_slow = MPBP(bp.g, [GenericFactor.(w) for w in bp.w], bp.ϕ, bp.ψ, 
            deepcopy(collect(bp.μ)), deepcopy(bp.b), collect(bp.f))
        iterate!(bp_slow; tol=1e-14, maxiter=10, damp=0.5, svd_trunc)
        m_bp_slow = [only(m) for m in means(f, bp_slow)]
        @test m_bp_slow ≈ m_bp
    end

end