rng = MersenneTwister(111)

T = 3
J = [0 1 0 0 0;
     1 0 1 1 0;
     0 1 0 0 0;
     0 1 0 0 0;
     0 0 0 0 0] .|> float

N = size(J, 1)
h = randn(rng, N)

β = 1.0
ising = Ising(J, h, β)

gl = Glauber(ising, T)

for i in 1:N
    r = 0.75
    gl.ϕ[i][1] .*= [r, 1-r]
end

bp2 = mpbp(gl)
w = [[UniformBPFactor(www) for www in ww] for ww in bp2.w]
bp = mpbp(bp2.g, w, fill(2,N), T)

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

@testset "UniformBPFactor" begin
    @test Z_exact ≈ Z_bp
    @test p_ex ≈ p_bp
    @test all(all(isapprox(p, 0.5) for p in pp) for pp in p_bp)
end