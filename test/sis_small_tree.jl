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
rng = MersenneTwister(111)
draw_node_observations!(bp, N, last_time=true; rng)

svd_trunc = TruncThresh(0.0)
iterate!(bp, maxiter=10; svd_trunc, showprogress=false)

b_bp = beliefs(bp)
p_bp = [[bbb[2] for bbb in bb] for bb in b_bp]

p_exact, Z_exact = exact_prob(bp)
b_exact = exact_marginals(bp; p_exact)
p_ex = [[bbb[2] for bbb in bb] for bb in b_exact]

f_bethe = bethe_free_energy(bp)

r_bp = autocorrelations(bp)
r_exact = exact_autocorrelations(bp)

c_bp = autocovariances(bp)
c_exact = exact_autocovariances(bp)

@testset "SIS small tree" begin
    @test isapprox(Z_exact, exp(-f_bethe), atol=1e-5)
    @test p_ex ≈ p_bp
    @test isapprox(r_bp, r_exact, atol=1e-3)
    @test isapprox(c_bp, c_exact, atol=1e-3)
end