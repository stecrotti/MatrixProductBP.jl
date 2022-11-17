q = q_sis
T = 3

A = [0 1 1 1; 1 0 0 0; 1 0 0 0; 1 0 0 0]
g = IndexedGraph(A)
N = size(A, 1)

λ = 0.2
ρ = 0.1
γ = 0.1

sis = SIS(g, λ, ρ, T; γ)
bp = mpbp(sis)
draw_node_observations!(bp, N, last_time=true)

svd_trunc = TruncThresh(0.0)
iterate!(bp, maxiter=10; svd_trunc, showprogress=false)

b_bp = beliefs(bp)
p_bp = [[bbb[2] for bbb in bb] for bb in b_bp]

p_exact, Z_exact = exact_prob(bp)
b_exact = site_time_marginals(bp; m = site_marginals(bp; p=p_exact))
p_ex = [[bbb[2] for bbb in bb] for bb in b_exact]

f_bethe = bethe_free_energy(bp)

@testset "SIS small tree" begin
    @test isapprox(Z_exact, exp(-f_bethe), atol=1e-5)
    @test p_ex ≈ p_bp
end