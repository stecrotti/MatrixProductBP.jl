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

svd_trunc = TruncBond(4)
iterate!(bp, maxiter=10; svd_trunc, showprogress=false, svd_verbose=true)

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
                    deepcopy(collect(bp.μ)), collect(bp.b), collect(bp.f))

    for i=1:20
        X, _ = onesample(bp; rng)
        @test logprob(bp, X) ≈ logprob(bpfake, X)
    end

    iterate!(bpfake, maxiter=10; svd_trunc, showprogress=false)

    @test beliefs(bpfake) ≈ beliefs(bp)

end

struct SlowFactor{F} <: BPFactor
    w::F
end

(w::SlowFactor)(xᵢᵗ⁺¹::Integer, xₙᵢᵗ::AbstractVector{<:Integer}, xᵢᵗ::Integer) = w.w(xᵢᵗ⁺¹, xₙᵢᵗ, xᵢᵗ)

@testset "SlowFactor - extensive trace update test" begin
    rng2 = MersenneTwister(111)
    bpfake = MPBP(bp.g, [SlowFactor.(w) for w in bp.w], bp.ϕ, bp.ψ, 
                    deepcopy(collect(bp.μ)), collect(bp.b), collect(bp.f))

    for i=1:20
        X, _ = onesample(bp; rng)
        @test logprob(bp, X) ≈ logprob(bpfake, X)
    end

    iterate!(bpfake, maxiter=10; svd_trunc, showprogress=false)

    @test beliefs(bpfake) ≈ beliefs(bp)
end

struct RecursiveTraceFactor{F<:BPFactor,N} <: RecursiveBPFactor
    w :: F
end

RecursiveTraceFactor(f,N) = RecursiveTraceFactor{typeof(f),N}(f)

nstates(::Type{RecursiveTraceFactor{F,N}}, d::Integer) where {F,N} = N^d

function prob_y(wᵢ::U, xᵢᵗ⁺¹, xᵢᵗ, yₙᵢᵗ, dᵢ) where {U <: RecursiveTraceFactor{F,N}} where {F,N}
    wᵢ.w(xᵢᵗ⁺¹, reverse(digits(yₙᵢᵗ-1; base=N, pad=dᵢ)) .+ 1, xᵢᵗ)
end

prob_xy(wᵢ::RecursiveTraceFactor, yₖ, xₖ, xᵢ, k) = (yₖ == xₖ)

prob_yy(wᵢ::U, y, y1, y2, xᵢ, d1, d2) where {U<:RecursiveTraceFactor} = (y - 1 == (y1 - 1)  + (y2 - 1) * d1)



@testset "RecursiveTraceFactor" begin
    rng2 = MersenneTwister(111)
    bpfake = MPBP(bp.g, [RecursiveTraceFactor.(w,2) for w in bp.w], bp.ϕ, bp.ψ, 
                    deepcopy(collect(bp.μ)), collect(bp.b), collect(bp.f))

    for i=1:20
        X, _ = onesample(bp; rng)
        @test logprob(bp, X) ≈ logprob(bpfake, X)
    end

    iterate!(bpfake, maxiter=10; svd_trunc=TruncBond(10), showprogress=false)

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