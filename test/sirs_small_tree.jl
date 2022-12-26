using MatrixProductBP, MatrixProductBP.Models, IndexedGraphs, Test, Random
import MatrixProductBP: nstates
import MatrixProductBP.Models: prob_xy, prob_yy, prob_y



T = 2

A = [0 1 1; 1 0 0; 1 0 0]
g = IndexedGraph(A)
N = size(A, 1)

λ = 0.5
ρ = 0.4
γ = 0.5
ρ = 0.2
σ = 0.1

sirs = SIRS(g, λ, ρ, σ, T; γ)
bp = mpbp(sirs)
rng = MersenneTwister(111)
X, _ = onesample(bp; rng)

@testset "logprob" begin
    @test logprob(bp, X) ≈ -3.912023005428146
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

f(x,i) = x == INFECTED

r_bp = autocorrelations(f, bp)
r_exact = exact_autocorrelations(f, bp; p_exact)

c_bp = autocovariances(f, bp)
c_exact = exact_autocovariances(f, bp; r = r_exact)

@testset "SIRS small tree" begin
    @test Z_exact ≈ Z_bp
    @test p_ex ≈ p_bp
    @test r_bp ≈ r_exact
    @test c_bp ≈ c_exact
end

struct FakeSIRS <: RecursiveBPFactor
    w::SIRSFactor
end

prob_xy(w::FakeSIRS, x...) = prob_xy(w.w, x...)
prob_yy(w::FakeSIRS, x...) = prob_yy(w.w, x...)
prob_y(w::FakeSIRS, x...) = prob_y(w.w, x...)

nstates(::Type{<:FakeSIRS}, l::Int) = nstates(SIRSFactor, l)

struct SlowSIRS <: BPFactor
    w::SIRSFactor
end

(w::SlowSIRS)(x...) = w.w(x...)


@testset "FakeSIRS - RecursiveBPFactor generic methods" begin
    rng2 = MersenneTwister(111)
    bpfake = MPBP(bp.g, [FakeSIRS.(w) for w in bp.w], bp.ϕ, bp.ψ, 
                    deepcopy(collect(bp.μ)), deepcopy(bp.b), collect(bp.f))
    bpslow = MPBP(bp.g, [SlowSIRS.(w) for w in bp.w], bp.ϕ, bp.ψ, 
                    deepcopy(collect(bp.μ)), deepcopy(bp.b), collect(bp.f))

    for i=1:20
        X, _ = onesample(bp; rng)
        @test logprob(bp, X) ≈ logprob(bpfake, X) ≈ logprob(bpslow, X)
    end

    iterate!(bpfake, maxiter=10; svd_trunc, showprogress=false)
    iterate!(bpslow, maxiter=10; svd_trunc, showprogress=false)

    @test beliefs(bpfake) ≈ beliefs(bp)
    @test beliefs(bpfake) ≈ beliefs(bpslow)

end




# observe everything and check that the free energy corresponds to the posterior of sample `X`
sirs = SIRS(g, λ, ρ, σ, T; γ)
draw_node_observations!(bp.ϕ, X, N*(T+1), last_time=false)
reset_messages!(bp)
iterate!(bp, maxiter=10; svd_trunc, showprogress=false)
f_bethe = bethe_free_energy(bp)
logl_bp = - f_bethe
logp = logprob(bp, X)

@testset "SIRS small tree - observe everything" begin
    @test logl_bp ≈ logp
end