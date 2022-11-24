```Glauber on a small tree, comparison with exact solution```
q = q_glauber
T = 3
J = [0 1 0 0;
     1 0 1 1;
     0 1 0 0;
     0 1 0 0] .|> float

N = size(J, 1)
h = randn(N)

β = 1.0
ising = Ising(J, h, β)

p⁰ = map(1:N) do i
    r = 0.75
    [r, 1-r]
end

gl = Glauber(ising, T; p⁰)
bp = mpbp(gl)

rng = MersenneTwister(111)
X = draw_node_observations!(bp, N; rng)

cb = CB_BP(bp; showprogress=false)
svd_trunc = TruncThresh(0.0)
iterate!(bp; maxiter=20, svd_trunc, cb)

b_bp = beliefs(bp)
p_bp = [[bbb[2] for bbb in bb] for bb in b_bp]

p_exact, Z_exact = exact_prob(bp)
b_exact = exact_marginals(bp; p_exact)
p_ex = [[bbb[2] for bbb in bb] for bb in b_exact]

f_bethe = bethe_free_energy(bp; svd_trunc)

r_bp = autocorrelations(bp)
r_exact = exact_autocorrelations(bp; p_exact)

c_bp = autocovariances(bp)
c_exact = exact_autocovariances(bp; r = r_exact)

@testset "Glauber small tree" begin
    @test isapprox(Z_exact, exp(-f_bethe); atol=1e-6)
    @test p_ex ≈ p_bp
    @test isapprox(r_bp, r_exact; atol=1e-6)
    @test isapprox(c_bp, c_exact; atol=1e-6)
end

# observe everything and check that the free energy corresponds to the prior of the sample `X`
draw_node_observations!(bp.ϕ, X, N*(T+1), last_time=false)
reset_messages!(bp)
iterate!(bp, maxiter=10; svd_trunc, showprogress=false)
f_bp = bethe_free_energy(bp)
logl_bp = -f_bp
logp, logl = logprior_loglikelihood(bp, X)

@testset "Glauber small tree - observe everything" begin
    @test isapprox(logl_bp, logp)
end