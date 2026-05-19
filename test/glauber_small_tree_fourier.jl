@testset "Glauber small tree (Fourier)" begin
    rng = MersenneTwister(111)
    atol = 1e-5
    rtol = 1e-2

    T = 2
    β = 1.0
    A = [0 1 0 0 0;
        1 0 1 1 0;
        0 1 0 0 0;
        0 1 0 0 0;
        0 0 0 0 0]

    N = size(A, 1)
    h = randn(rng, N)

    J = randn(rng, N, N) .* A
    J = 0.5 * (J + J')  # make symmetric

    ising = Ising(J, h, β)
    gl = Glauber(ising, T)

    for i in 1:N
        r = 0.75
        gl.ϕ[i][1] .*= [r, 1-r]
    end

    bp = mpbp(deepcopy(gl))
    X, observed = draw_node_observations!(bp, N; rng)
    bp_fourier = mpbp_fourier(deepcopy(bp), K=300, σ=1/300)

    # svd_trunc = TruncBond(10)
    # cb_fourier = CB_BP(bp_fourier; showprogress=false, info="Glauber Fourier")
    # iterate!(bp_fourier; maxiter=20, svd_trunc, cb=cb_fourier)

    # b_bp_fourier = beliefs(bp_fourier)
    # p_bp_fourier = [[bbb[2] for bbb in bb] for bb in b_bp_fourier]

    # p_exact, Z_exact = exact_prob(bp)
    # b_exact = exact_marginals(bp; p_exact)
    # p_ex = [[bbb[2] for bbb in bb] for bb in b_exact]

    # f_bethe_fourier = bethe_free_energy(bp_fourier)
    # Z_bp_fourier = exp(-f_bethe_fourier)

    # f(x,i) = potts2spin(x,i)

    # r_bp_fourier = autocorrelations(f, bp_fourier)
    # r_exact = exact_autocorrelations(f, bp; p_exact)

    # c_bp_fourier = autocovariances(f, bp_fourier)
    # c_exact = exact_autocovariances(f, bp; r = r_exact)

    # pb_bp_fourier = pair_beliefs(bp_fourier)[1]
    # pb_bp2_fourier = marginals.(pair_beliefs_as_mpem(bp_fourier)[1])

    # @testset "Observables" begin
    #     @test isapprox(Z_exact, Z_bp_fourier; rtol, atol)
    #     for (_p_ex, _p_bp) in zip(p_ex, p_bp_fourier)
    #         @test isapprox(_p_ex, _p_bp; rtol, atol)
    #     end
    #     @test isapprox(r_exact, r_bp_fourier; rtol, atol)
    #     for (_c_ex, _c_bp) in zip(c_exact, c_bp_fourier)
    #         @test isapprox(_c_ex, _c_bp; rtol, atol)
    #     end
    #     for (_pb_bp, _pb_bp2) in zip(pb_bp_fourier, pb_bp2_fourier)
    #         for (_pb_bp_mat, _pb_bp2_mat) in zip(_pb_bp, _pb_bp2)
    #             @test isapprox(_pb_bp_mat, _pb_bp2_mat; rtol, atol)
    #         end
    #     end
    # end

    # # observe everything and check that the free energy corresponds to the posterior of sample `X`
    # reset!(bp; observations=true)
    # draw_node_observations!(bp.ϕ, X, N*(T+1), last_time=false)
    # reset_messages!(bp)
    # bp_fourier = mpbp_fourier(deepcopy(bp), K=300, σ=1/300)
    # cb = CB_BP(bp_fourier; showprogress=false)
    # iters, cb = iterate!(bp_fourier, maxiter=50; svd_trunc, showprogress=false, tol=1e-8)
    # f_bethe = bethe_free_energy(bp_fourier)
    # logl_bp = -f_bethe
    # logp = logprob(bp_fourier, X)

    # @testset "Glauber small tree - observe everything" begin
    #     @test isapprox(logl_bp, logp; rtol, atol)
    # end

    # test DampedFactor
    p = 0.2

    J = [0 1 0 0 0;
        1 0 1 1 0;
        0 1 0 0 0;
        0 1 0 0 0;
        0 0 0 0 0] .|> float
    
    ising = Ising(J, h, β)
    gl = Glauber(ising, T)

    for i in 1:N
        r = 0.75
        gl.ϕ[i][1] .*= [r, 1-r]
    end

    bp = mpbp(deepcopy(gl))
    w = [[DampedFactor(www, p) for www in ww] for ww in bp.w]

    # w = [BPFactor[typeof(www)<:RecursiveBPFactor ? DampedFactor(www,p) : DampedGGFactor(www, p) for www in ww] for ww in bp.w]
    bp2 = mpbp(bp.g, w, fill(2,N), T; ϕ = bp.ϕ)
    X, observed = draw_node_observations!(bp2, N; rng)

    w_fourier = [fill(FourierGlauberFactor(convert(Vector{Float64}, collect([J[ed.src,ed.dst] for ed in inedges(bp.g,i)])), h[i], β; σ=1/300, K=300, p=p), T+1) for i in vertices(bp.g)]
    bp_fourier = mpbp(ComplexF64, bp.g, w_fourier, fill(2, nv(bp.g)), T; ϕ=bp2.ϕ)

    svd_trunc = TruncBondThresh(10)
    cb = CB_BP(bp_fourier; showprogress=false, info="Glauber")
    iterate!(bp_fourier; maxiter=20, svd_trunc, cb)

    b_bp_fourier = beliefs(bp_fourier)
    p_bp_fourier = [[bbb[2] for bbb in bb] for bb in b_bp_fourier]

    p_exact, Z_exact = exact_prob(bp2)
    b_exact = exact_marginals(bp_fourier; p_exact)
    p_ex = [[bbb[2] for bbb in bb] for bb in b_exact]

    f_bethe_fourier = bethe_free_energy(bp_fourier)
    Z_bp_fourier = exp(-f_bethe_fourier)

    f(x,i) = 2x-3

    r_bp_fourier = autocorrelations(f, bp_fourier)
    r_exact = exact_autocorrelations(f, bp2; p_exact)

    c_bp_fourier = autocovariances(f, bp_fourier)
    c_exact = exact_autocovariances(f, bp2; r = r_exact)

    
    pb_bp_fourier = pair_beliefs(bp_fourier)[1]
    pb_bp2_fourier = marginals.(pair_beliefs_as_mpem(bp_fourier)[1])

    @testset "Glauber small tree - DampedFactor" begin
        @test isapprox(Z_exact, Z_bp_fourier; rtol, atol)
        for (_p_ex, _p_bp) in zip(p_ex, p_bp_fourier)
            @test isapprox(_p_ex, _p_bp; rtol, atol)
        end
        @test isapprox(r_exact, r_bp_fourier; rtol, atol)
        for (_c_ex, _c_bp) in zip(c_exact, c_bp_fourier)
            @test isapprox(_c_ex, _c_bp; rtol, atol)
        end
        for (_pb_bp, _pb_bp2) in zip(pb_bp_fourier, pb_bp2_fourier)
            for (_pb_bp_mat, _pb_bp2_mat) in zip(_pb_bp, _pb_bp2)
                @test isapprox(_pb_bp_mat, _pb_bp2_mat; rtol, atol)
            end
        end
    end



    # test DampedFactor
    reset!(bp; observations=true)
    X, observed = draw_node_observations!(bp, N; rng)
    p = 0.2
    w = [[DampedFactor(www, p) for www in ww] for ww in bp.w]
    bp2 = mpbp(bp.g, w, fill(2,N), T; ϕ = bp.ϕ)

    svd_trunc = TruncBondThresh(10)
    cb = CB_BP(bp2; showprogress=false, info="Glauber")
    iterate!(bp2; maxiter=20, svd_trunc, cb)

    b_bp = beliefs(bp2)
    p_bp = [[bbb[2] for bbb in bb] for bb in b_bp]

    p_exact, Z_exact = exact_prob(bp2)
    b_exact = exact_marginals(bp2; p_exact)
    p_ex = [[bbb[2] for bbb in bb] for bb in b_exact]

    f_bethe = bethe_free_energy(bp2)
    Z_bp = exp(-f_bethe)

    local f(x,i) = 2x-3

    r_bp = autocorrelations(f, bp2)
    r_exact = exact_autocorrelations(f, bp2; p_exact)

    c_bp = autocovariances(f, bp2)
    c_exact = exact_autocovariances(f, bp2; r = r_exact)

    pb_bp = pair_beliefs(bp2)[1]
    pb_exact = exact_pair_marginals(bp2)

    a_bp = alternate_marginals(bp2)
    a_exact = exact_alternate_marginals(bp2)


    @testset "Glauber small tree - DampedFactor" begin
        @test Z_exact ≈ Z_bp
        @test p_ex ≈ p_bp
        @test r_bp ≈ r_exact
        @test c_bp ≈ c_exact
        @test pb_bp ≈ pb_exact
        @test a_bp ≈ a_exact
    end
end
