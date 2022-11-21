```Glauber on a small tree, comparison with exact solution```
q = q_glauber
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
    r = 0.75
    [r, 1-r]
end

gl = Glauber(ising, T; p⁰)
bp = mpbp(gl)

draw_node_observations!(bp, N)

cb = CB_BP(bp; showprogress=false)
svd_trunc = TruncThresh(0.0)
iterate!(bp, maxiter=10; svd_trunc, cb)

b_bp = beliefs(bp)
p_bp = [[bbb[2] for bbb in bb] for bb in b_bp]

p_exact, Z_exact = exact_prob(bp)
b_exact = exact_marginals(bp; m = site_marginals(bp; p=p_exact))
p_ex = [[bbb[2] for bbb in bb] for bb in b_exact]

f_bethe = bethe_free_energy(bp)

@testset "Glauber small tree" begin
    @test isapprox(Z_exact, exp(-f_bethe))
    @test p_ex ≈ p_bp
end