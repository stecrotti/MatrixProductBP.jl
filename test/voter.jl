@testset "Voter model" begin
    rng = MersenneTwister(111)

    T = 3

    J = [0 -1  0  0;
        -1 0  1  1;
        0  1  0  0;
        0  1  0  0] .|> float 

    N = size(J, 1)
    h = randn(rng, N)

    β = 2.0
    
    w = map(1:N) do i
        wᵢᵗ = VoterFactor([J[i,j] for j in 1:N if J[i,j]!= 0])
        fill(wᵢᵗ, T + 1)
    end

    bp = mpbp(IndexedBiDiGraph(J), w, fill(2,N), T)

    for i in 1:N
        r = 0.75
        bp.ϕ[i][1] .*= [r, 1-r]
    end

    @testset "Factor type" begin
        @test all(eltype(wᵢ) <: VoterFactor for wᵢ in bp.w)
    end

    X = draw_node_observations!(bp, N; rng)

    svd_trunc = TruncThresh(0.0)
    cb = CB_BP(bp; showprogress=false)
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

    @testset "Observables" begin
        @test Z_exact ≈ Z_bp
        @test p_ex ≈ p_bp
        @test r_bp ≈ r_exact
        @test c_bp ≈ c_exact
    end
end