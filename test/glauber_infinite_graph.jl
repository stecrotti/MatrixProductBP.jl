""" Compute marginals and free energy on an infinite k-regular graph, with observations.
Compare with results on a complete (k+1)-graph, which to BP is indistinguishable from the infinite k-regular.
This tests the computation of the free energy on the infinite graph """

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

iters, cb = iterate!(bp; maxiter=100, svd_trunc=TruncThresh(0.0), cb, tol=1e-15, damp=0.1)

b_bp = beliefs(bp)
p_bp = [[bbb[2] for bbb in bb] for bb in b_bp]

f_bethe = bethe_free_energy(bp)
Z_bp = exp(-f_bethe)

N = k+1
g = IndexedBiDiGraph(complete_graph(N))
bp_exact = mpbp(g, fill(wᵢ, N), fill(2,N), T)
for i in 1:N; bp_exact.ϕ[i] = ϕᵢ; end

iterate!(bp_exact; maxiter=100, svd_trunc=TruncThresh(0.0), cb, tol=1e-15)

b_exact = beliefs(bp_exact)
p_exact = [[bbb[2] for bbb in bb] for bb in b_exact][1:1]

f_bethe_exact = bethe_free_energy(bp_exact)
Z_exact = exp(-1/N*f_bethe_exact)

@testset "Glauber infinite graph" begin
    @test Z_exact ≈ Z_bp
    @test p_exact ≈ p_bp
end