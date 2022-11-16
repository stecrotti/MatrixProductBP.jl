T = 3
q = q_glauber

J = [0 1 0 0 0;
     1 0 1 0 0;
     0 1 0 1 1;
     0 0 1 0 0;
     0 0 1 0 0] .|> float
for ij in eachindex(J)
    if J[ij] !=0 
        J[ij] = randn()
    end
end
J = J + J'
g = IndexedGraph(J)

N = 5
h = randn(N)

β = 1.0

p⁰ = map(1:N) do i
    r = rand()
    [r, 1-r]
end
ϕ = [[ones(2) for t in 1:T] for i in 1:N]

O = [ (1, 2, 1, [0.1 0.9; 0.3 0.4]),
      (3, 4, 2, [0.4 0.6; 0.5 0.9]),
      (3, 5, 2, rand(2,2)),
      (2, 3, T, rand(2,2))]

ψ = pair_observations_nondirected(O, g, T, q)

ising = Ising(J, h, β)
gl = Glauber(ising, p⁰, ϕ, ψ)

bp = mpbp(gl)
iterate!(bp)

z_msg = [normalization(A) for A in bp.μ]

@testset "Message normaliz" begin
    for A in bp.μ
        normalize!(A)
        @test normalization(A) ≈ 1
    end
end

