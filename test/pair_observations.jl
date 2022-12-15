```Glauber on a small tree with pair observations, comparison with exact solution```

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

O = [ (1, 2, 1, [0.1 0.9; 0.3 0.4]),
      (3, 4, 2, [0.4 0.6; 0.5 0.9]),
      (3, 5, 2, rand(2,2)) ,
      (2, 3, T, rand(2,2))          ]

ψ = pair_observations_nondirected(O, ising.g, T, 2)

gl = Glauber(ising, T; ψ)

for i in 1:N
    r = 0.15
    gl.ϕ[i][1] .* [r, 1-r]
end

bp = mpbp(gl)

draw_node_observations!(bp, N)

cb = CB_BP(bp; showprogress=false)
svd_trunc = TruncThresh(0.0)
iterate!(bp, maxiter=10; svd_trunc, cb, showprogress=false)

b_bp = beliefs(bp)
p_bp = [[bbb[2] for bbb in bb] for bb in b_bp]

p_exact, Z_exact = exact_prob(bp)
b_exact = exact_marginals(bp)
p_ex = [[bbb[2] for bbb in bb] for bb in b_exact]

f_bethe = bethe_free_energy(bp; svd_trunc)
Z_bp = exp(-f_bethe)

r_bp = autocorrelations(bp)
r_exact = exact_autocorrelations(bp)

@testset "Pair observations - SimpleBPFactor" begin
    @test Z_exact ≈ Z_bp
    @test p_ex ≈ p_bp
    @test r_bp ≈ r_exact
end

###################
# Now check generic BP
for ij in eachindex(J)
    if J[ij] !=0 
        J[ij] = randn()
    end
end
J = J + J'

N = size(J, 1)
h = randn(N)

β = 1.0
ising = Ising(J, h, β)

O = [ (1, 2, 1, [0.1 0.9; 0.3 0.4]),
      (3, 4, 2, [0.4 0.6; 0.5 0.9]),
      (3, 5, 2, rand(2,2)) ,
      (2, 3, T, rand(2,2))          ]

ψ = pair_observations_nondirected(O, ising.g, T, 2)

gl = Glauber(ising, T; ψ)

for i in 1:N
    r = 0.15
    gl.ϕ[i][1] .* [r, 1-r]
end

bp = mpbp(gl)

draw_node_observations!(bp, N)

cb = CB_BP(bp; showprogress=false)
svd_trunc = TruncThresh(0.0)
iterate!(bp, maxiter=10; svd_trunc, cb, showprogress=false)

b_bp = beliefs(bp)
p_bp = [[bbb[2] for bbb in bb] for bb in b_bp]

p_exact, Z_exact = exact_prob(bp)
b_exact = exact_marginals(bp)
p_ex = [[bbb[2] for bbb in bb] for bb in b_exact]

f_bethe = bethe_free_energy(bp; svd_trunc)
Z_bp = exp(-f_bethe)

r_bp = autocorrelations(bp)
r_exact = exact_autocorrelations(bp)

@testset "Pair observations - generic BP" begin
    @test Z_exact ≈ Z_bp
    @test p_ex ≈ p_bp
    @test r_bp ≈ r_exact
end