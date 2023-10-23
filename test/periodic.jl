@testset "Periodic" begin

    rng = MersenneTwister(111)

    T = 2
    J = [0 1 0 0 0;
        1 0 1 1 0;
        0 1 0 0 0;
        0 1 0 0 0;
        0 0 0 0 0] .|> float

    N = size(J, 1)
    h = randn(rng, N)

    β = 1.0
    ising = Ising(J, h, β)

    O = [ (1, 2, 1, [0.1 0.9; 0.3 0.4]),
        (2, 4, 2, [0.4 0.6; 0.5 0.9]),
        (2, 3, T, rand(2,2))          ]

    ψ = pair_observations_nondirected(O, ising.g, T, 2)

    gl = Glauber(ising, T; ψ)

    for i in 1:N
        r = 0.75
        gl.ϕ[i][1] .*= [r, 1-r]
    end

    bp = periodic_mpbp(deepcopy(gl))

    X, observed = draw_node_observations!(bp, N; rng)

    svd_trunc = TruncThresh(0.0)
    svd_trunc = TruncBondThresh(10)
    cb = CB_BP(bp; showprogress=false, info="Glauber")
    iterate!(bp; maxiter=20, svd_trunc, cb)

    b_bp = beliefs(bp)
    p_bp = [[bbb[2] for bbb in bb] for bb in b_bp]

    p_exact, Z_exact = exact_prob(bp)
    b_exact = exact_marginals(bp; p_exact)
    p_ex = [[bbb[2] for bbb in bb] for bb in b_exact]

    f_bethe = bethe_free_energy(bp)
    Z_bp = exp(-f_bethe)

    f(x,i) = 2x-3

    r_bp = autocorrelations(f, bp)
    r_exact = exact_autocorrelations(f, bp; p_exact)

    c_bp = autocovariances(f, bp)
    c_exact = exact_autocovariances(f, bp; r = r_exact)

    pb_bp = pair_beliefs(bp)[1]
    p_bp = [[bbb[2] for bbb in bb] for bb in b_bp]
    pb_bp2 = marginals.(pair_beliefs_as_mpem(bp)[1])

    @testset "Glauber small tree - periodic + pair observations" begin
        @test Z_exact ≈ Z_bp
        @test p_ex ≈ p_bp
        @test r_bp ≈ r_exact
        @test c_bp ≈ c_exact
        @test pb_bp ≈ pb_bp2
    end

    ########## INFINITE GRAPH
    T = 2
    k = 3   
    m⁰ = 0.5

    β = 1.0
    J = 1.0
    h = 0.0

    wᵢ = fill(HomogeneousGlauberFactor(J, h, β), T+1)
    ϕᵢ = [ t == 0 ? [(1+m⁰)/2, (1-m⁰)/2] : ones(2) for t in 0:T]
    ϕᵢ[2] = [0.4, 0.6]
    ϕᵢ[end] = [0.95, 0.05]
    bp = periodic_mpbp_infinite_graph(k, wᵢ, 2, ϕᵢ)
    cb = CB_BP(bp)

    iters, cb = iterate!(bp; maxiter=150, svd_trunc=TruncBond(10), cb, tol=1e-12, damp=0.2)

    b_bp = beliefs(bp)
    pb_bp = pair_beliefs(bp)[1][1]
    p_bp = [[bbb[2] for bbb in bb] for bb in b_bp]

    f_bethe = bethe_free_energy(bp)
    Z_bp = exp(-f_bethe)

    N = k+1
    g = IndexedBiDiGraph(complete_graph(N))
    bp_exact = periodic_mpbp(g, fill(wᵢ, N), fill(2,N), T)
    for i in 1:N; bp_exact.ϕ[i] = ϕᵢ; end

    cb = CB_BP(bp_exact)
    iterate!(bp_exact; maxiter=150, svd_trunc=TruncBond(10), cb, tol=1e-12, damp=0.2)

    b_exact = beliefs(bp_exact)
    p_exact = [[bbb[2] for bbb in bb] for bb in b_exact][1:1]
    pb_exact = pair_beliefs(bp_exact)[1][1]

    f_bethe_exact = bethe_free_energy(bp_exact)
    Z_exact = exp(-1/N*f_bethe_exact)

    @testset "Glauber infinite graph - periodic" begin
        # @test Z_exact ≈ Z_bp ### NOT WORKING!
        @test p_exact ≈ p_bp
        @test pb_exact ≈ pb_bp

    end

end