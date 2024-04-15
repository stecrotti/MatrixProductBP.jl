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
    λ = sparse(λ)
    ρ = fill(ρ_unif,N)
    γ = 0.13

    sis_u = SIS(g, λ_unif, ρ_unif, T; γ)
    sis_h = SIS_heterogeneous(λ, ρ, T; γ)


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


    A_ = copy(A)
    A_[3,2] = 0
    A_[4,3] = 0
    λ_ = copy(λ)
    λ_[3,2] = 0.0
    λ_[4,3] = 0.0

    sis_u_ = SIS(g, λ_unif, ρ_unif, T; γ)
    sis_h_ = SIS_heterogeneous(λ, ρ, T; γ)


    bp_u_ = mpbp(sis_u_)
    iters_u_, cb_u_ = iterate!(bp_u_, maxiter=200; svd_trunc, tol=1e-12)
    b_bp_u_ = beliefs(bp_u_)
    p_bp_u_ = [[bᵗ[INFECTIOUS] for bᵗ in bb] for bb in b_bp_u_]

    bp_h_ = mpbp(sis_h_)
    iters_h_, cb_h_ = iterate!(bp_h_, maxiter=200; svd_trunc, tol=1e-12)
    b_bp_h_ = beliefs(bp_h_)
    p_bp_h_ = [[bᵗ[INFECTIOUS] for bᵗ in bb] for bb in b_bp_h_]

    err_ = maximum([(p_bp_h_[i][j]-p_bp_u_[i][j])/p_bp_u_[i][j] for i in eachindex(p_bp_u_) for j in eachindex(p_bp_u_[i])])

    @test err_ ≈ 0

end