using MatrixProductBP, MatrixProductBP.Models, IndexedGraphs, Test, Random
import MatrixProductBP: nstates
import MatrixProductBP.Models: prob_xy, prob_yy, prob_y



T = 3

A = [0 1 1 1; 1 0 0 0; 1 0 0 0; 1 0 0 0]
g = IndexedGraph(A)
N = size(A, 1)

λ = 0.5
ρ = 0.4
γ = 0.5

sis = SIS(g, λ, ρ, T; γ)
bp = mpbp(sis)
rng = MersenneTwister(111)
X, _ = onesample(bp; rng)

@testset "logprob" begin
    @test logprob(bp, X) ≈ -8.50477067141768 
end

draw_node_observations!(bp.ϕ, X, N, last_time=true; rng)

svd_trunc = TruncThresh(0.0)
iterate!(bp, maxiter=10; svd_trunc, showprogress=false)

b_bp = beliefs(bp)
p_bp = [[bbb[2] for bbb in bb] for bb in b_bp]

p_exact, Z_exact = exact_prob(bp)
b_exact = exact_marginals(bp; p_exact)
p_ex = [[bbb[2] for bbb in bb] for bb in b_exact]

f_bethe = bethe_free_energy(bp)
Z_bp = exp(-f_bethe)

f(x,i) = x-1

r_bp = autocorrelations(f, bp)
r_exact = exact_autocorrelations(f, bp; p_exact)

c_bp = autocovariances(f, bp)
c_exact = exact_autocovariances(f, bp; r = r_exact)

@testset "SIS small tree" begin
    @test Z_exact ≈ Z_bp
    @test p_ex ≈ p_bp
    @test r_bp ≈ r_exact
    @test c_bp ≈ c_exact
end

struct FakeSIS <: RecursiveBPFactor
    w::SISFactor
end

prob_xy(w::FakeSIS, x...) = prob_xy(w.w, x...)
prob_yy(w::FakeSIS, x...) = prob_yy(w.w, x...)
prob_y(w::FakeSIS, x...) = prob_y(w.w, x...)

nstates(::Type{<:FakeSIS}, l::Int) = nstates(SISFactor, l)


@testset "FakeSIS - RecursiveBPFactor generic methods" begin
    rng2 = MersenneTwister(111)
    bpfake = MPBP(bp.g, [FakeSIS.(w) for w in bp.w], bp.ϕ, bp.ψ, 
                    deepcopy(bp.μ), deepcopy(bp.b), deepcopy(bp.f))

    for i=1:20
        X, _ = onesample(bp; rng)
        @test logprob(bp, X) ≈ logprob(bpfake, X)
    end

    iterate!(bpfake, maxiter=10; svd_trunc, showprogress=false)

    @test beliefs(bpfake) ≈ beliefs(bp)

end




# observe everything and check that the free energy corresponds to the posterior of sample `X`
sis = SIS(g, λ, ρ, T; γ)
draw_node_observations!(bp.ϕ, X, N*(T+1), last_time=false)
reset_messages!(bp)
iterate!(bp, maxiter=10; svd_trunc, showprogress=false)
f_bethe = bethe_free_energy(bp)
logl_bp = - f_bethe
logp = logprob(bp, X)

@testset "SIS small tree - observe everything" begin
    @test logl_bp ≈ logp
end