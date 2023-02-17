```Glauber on a small tree, comparison with exact solution```

rng = MersenneTwister(111)

T = 3
J = [0 1 0 0;
     1 0 1 1;
     0 1 0 0;
     0 1 0 0] .|> float

N = size(J, 1)
h = randn(rng, N)

β = 1.0
ising = Ising(J, h, β)

gl = Glauber(ising, T)

for i in 1:N
    r = 0.75
    gl.ϕ[i][1] .*= [r, 1-r]
end

bp = mpbp(gl)

X, observed = draw_node_observations!(bp, N; rng)

svd_trunc = TruncThresh(0.0)
svd_trunc = TruncThreshBond(0.0, 10)
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

@testset "Glauber small tree" begin
    @test Z_exact ≈ Z_bp
    @test p_ex ≈ p_bp
    @test r_bp ≈ r_exact
    @test c_bp ≈ c_exact
end

# observe everything and check that the free energy corresponds to the posterior of sample `X`
reset!(bp)
draw_node_observations!(bp.ϕ, X, N*(T+1), last_time=false)
reset_messages!(bp)
cb = CB_BP(bp; showprogress=false)
iters, cb = iterate!(bp, maxiter=50; svd_trunc, showprogress=false, tol=0)
f_bethe = bethe_free_energy(bp)
logl_bp = - f_bethe
logp = logprob(bp, X)

@testset "Glauber small tree - observe everything" begin
    @test logl_bp ≈ logp
end