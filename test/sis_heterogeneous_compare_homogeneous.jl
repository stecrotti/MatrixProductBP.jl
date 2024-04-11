@testset "SIS heterogeneous compare homogeneous" begin
    T = 3
    N = 5
    
    A = [0 1 1 0 0;
         1 0 1 0 0;
         1 1 0 1 0;
         0 0 1 0 1;
         0 0 0 1 0]
    g = IndexedGraph(A)

    λ_unif = 0.15
    ρ_unif = 0.12
    λ = λ_unif .* A
    ρ = fill(ρ_unif,N)
    γ = 0.13

    sis_u = SIS(g, λ_unif, ρ_unif, T; γ)
    sis_h = SIS_heterogeneous(g, λ, ρ, T; γ)

    
    svd_trunc = TruncBond(3)

    bp_u = mpbp(sis_u)
    iters_u, cb_u = iterate!(bp_u, maxiter=200; svd_trunc, tol=1e-12)
    b_bp_u = beliefs(bp_u)
    p_bp_u = [[bᵗ[INFECTIOUS] for bᵗ in bb] for bb in b_bp_u]

    bp_h = mpbp(sis_h)
    iters_h, cb_h = iterate!(bp_h, maxiter=200; svd_trunc, tol=1e-12)
    b_bp_h = beliefs(bp_h)
    p_bp_h = [[bᵗ[INFECTIOUS] for bᵗ in bb] for bb in b_bp_h]

    err = maximum([(p_bp_h[i][j]-p_bp_u[i][j])/p_bp_u[i][j] for i in eachindex(p_bp_u) for j in eachindex(p_bp_u[i])])

    @test err ≈ 0
end