""" Compute marginals and free energy on an infinite k-regular graph, with observations.
Compare with results on a complete (k+1)-graph, which to BP is indistinguishable from the infinite k-regular.
This tests the computation of the free energy on the infinite graph """


@testset "Glauber infinite graph" begin
    T = 3   
    k = 3   
    m⁰ = 0.5

    β = 1.0
    J = 1.0
    h = 0.0

    wᵢ = fill(HomogeneousGlauberFactor(J, h, β), T+1)
    ϕᵢ = [ t == 0 ? [(1+m⁰)/2, (1-m⁰)/2] : ones(2) for t in 0:T]
    ϕᵢ[2] = [0.4, 0.6]
    ϕᵢ[end] = [0.95, 0.05]
    bp = mpbp_infinite_graph(k, wᵢ, 2, ϕᵢ)
    cb = CB_BP(bp)

    iters, cb = iterate!(bp; maxiter=150, svd_trunc=TruncThresh(0.0), cb, tol=1e-15, damp=0.1)

    b_bp = beliefs(bp)
    pb_bp = pair_beliefs(bp)
    p_bp = [[bbb[2] for bbb in bb] for bb in b_bp]

    f_bethe = bethe_free_energy(bp)
    Z_bp = exp(-f_bethe)

    N = k+1
    g = IndexedBiDiGraph(complete_graph(N))
    bp_exact = mpbp(g, fill(wᵢ, N), fill(2,N), T)
    for i in 1:N; bp_exact.ϕ[i] = ϕᵢ; end

    iterate!(bp_exact; maxiter=150, svd_trunc=TruncThresh(0.0), cb, tol=1e-15)

    b_exact = beliefs(bp_exact)
    p_exact = [[bbb[2] for bbb in bb] for bb in b_exact][1:1]

    f_bethe_exact = bethe_free_energy(bp_exact)
    Z_exact = exp(-1/N*f_bethe_exact)

    @test Z_exact ≈ Z_bp
    @test p_exact ≈ p_bp
end

@testset "Glauber infinite bipartite graph" begin
    T = 3   
    k = (3, 2)  
    m⁰ = 0.5

    β = 1.0
    JA = 1.0
    JB = -0.2
    h = -0.1

    w = [
        fill(HomogeneousGlauberFactor(JA, h, β), T+1),
        fill(HomogeneousGlauberFactor(JB, h, β), T+1)
    ]
    ϕ = [[t == 0 ? [(1+m⁰)/2, (1-m⁰)/2] : ones(2) for t in 0:T] for i in 1:2]
    ϕ[1][2] = [0.4, 0.6]
    ϕ[2][end] = [0.95, 0.05]
    bp = mpbp_infinite_bipartite_graph(k, w, (2, 2), ϕ)
    cb = CB_BP(bp)

    iters, cb = iterate!(bp; maxiter=150, svd_trunc=TruncThresh(0.0), cb, tol=1e-15, damp=0.1)

    b_bp = beliefs(bp)
    pb_bp = pair_beliefs(bp)
    p_bp = [[bbb[2] for bbb in bb] for bb in b_bp]

    f_bethe = bethe_free_energy(bp)
    Z_bp = exp(-f_bethe)

    N = sum(k)
    g = IndexedBiDiGraph(complete_bipartite_graph(reverse(k)...))
    w_exact = map(vertices(g)) do i
        fill(HomogeneousGlauberFactor(i ≤ k[2] ? β*JA : β*JB, β*h), T+1)
    end
    bp_exact = mpbp(g, w_exact, fill(2,N), T)
    for i in 1:N
        if i ≤ k[2]
            bp_exact.ϕ[i] = ϕ[1]
        else
            bp_exact.ϕ[i] = ϕ[2]
        end
    end

    iterate!(bp_exact; maxiter=150, svd_trunc=TruncThresh(0.0), cb, tol=1e-15)

    b_exact = beliefs(bp_exact)
    p_exact = [[bbb[2] for bbb in bb] for bb in b_exact][[1,k[1]+1]]

    f_bethe_exact = bethe_free_energy(bp_exact)
    Z_exact = exp(-1/N*f_bethe_exact)

    @test Z_exact ≈ Z_bp
    @test p_exact ≈ p_bp
end