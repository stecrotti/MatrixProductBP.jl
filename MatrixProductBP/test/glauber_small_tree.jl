```Glauber on a small tree, comparison with exact solution```

T = 2
J = [0 1 0 0 0;
     1 0 1 0 0;
     0 1 0 1 1;
     0 0 1 0 0;
     0 0 1 0 0] .|> float

N = size(J, 1)
h = randn(N)

β = 1.0
ising = Ising(J, h, β)

p⁰ = map(1:N) do i
    r = rand()
    r = 0.15
    [r, 1-r]
end
ϕ = [[ones(2) for t in 1:T] for i in 1:N]
ψ = [[ones(2,2) for t in 1:T] for _ in 1:ne(ising.g)]
gl = Glauber(ising, p⁰, ϕ, ψ)
bp = mpbp(gl)

svd_trunc = TruncBond(4)

draw_node_observations!(bp, N)
cb = CB_BP(bp; showprogress=false)
iterate!(bp, maxiter=10; svd_trunc, cb)

b_bp = beliefs(bp)
p_bp = [[bbb[2] for bbb in bb] for bb in b_bp]

p_exact, Z_exact = exact_prob(gl)
b_exact = site_time_marginals(gl; m = site_marginals(gl; p=p_exact))
p_ex = [[bbb[2] for bbb in bb] for bb in b_exact]

f_bethe = bethe_free_energy(bp)

@testset "Glauber small tree" begin
    @test Z_exact ≈ exp(-f_bethe)
    @test p_ex ≈ p_bp
end