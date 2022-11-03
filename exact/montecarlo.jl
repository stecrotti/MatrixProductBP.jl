import Distributions: sample!, sample
using Measurements
import Statistics: mean, std
import UnPack: @unpack
import Distributions: sample, Bernoulli
import Base.Threads: @threads
using Unzip
import StatsBase
import LogExpFunctions: logistic

include("../glauber.jl")
include("../mpdbp.jl")

function sweep!(x, ising::Ising; nodes=1:nv(ising.g))
    for i in nodes
        ∂i = neighbors(ising.g, i)
        p = local_w(ising.g, ising.J, ising.h, i, 1, x[∂i], ising.β)
        r = rand()
        if r < p
            x[i] = 1
        else
            x[i] = 2
        end
    end
    return x
end
sweep(x, ising::Ising; kw...) = sweep!(copy(x), ising; kw...)

# sample from ising (no dynamics)
struct EquilibriumGibbsSampler{T}
    ising :: Ising{T}
    X     :: Vector{Vector{Int}}

    function EquilibriumGibbsSampler(ising::Ising{T}) where {T<:AbstractFloat}
        X = Vector{Int64}[]
        new{T}(ising, X)
    end
end

function sample!(gs::EquilibriumGibbsSampler; nsamples = nv(gs.ising.g),
        x0 = rand(1:2, nv(gs.ising.g)),
        ntherm = 100)
    x = copy(x0)
    for _ in 1:ntherm
        sweep!(x, ising)
    end
    for _ in 1:nsamples
        sweep!(x, ising)
        push!(gs.X, copy(x))
    end
    return gs
end
sample(ising::Ising; kw...) = sample!(EquilibriumGibbsSampler(ising); kw...)

function magnetizations(gs::EquilibriumGibbsSampler) 
    m = potts2spin.( mean(gs.X) )
    s = std(gs.X) / sqrt(length(gs.X))
    m .± s
end

struct GlauberSampler{T,N,F}
    gl :: Glauber{T,N,F}
    X :: Vector{Matrix{Int}}

    function GlauberSampler(gl::Glauber{T,N,F}) where {T,N,F}
        X = Matrix{Int}[]
        new{T,N,F}(gl, X)
    end
end

# return a sampled trajectory
function sample!(X, gl::Glauber{T,N,F}) where {T,N,F}
    @assert is_free_dynamics(gl)
    @unpack ising = gl

    # t=0
    for i in 1:N
        p = p⁰[i][2]
        X[1,i] = rand(Bernoulli(p)) + 1
    end

    for t in 1:T
        for i in 1:N
            ∂i = neighbors(ising.g, i)
            p = local_w(ising.g, ising.J, ising.h, i, 2, 
                    X[t,∂i], ising.β)
            X[t+1,i] = rand(Bernoulli(p)) + 1
        end
    end
    X
end
function _sample(gl::Glauber{T,N,F}) where {T,N,F}
    X = zeros(Int, T+1, N)
    sample!(X, gl)
end

function sample!(gs::GlauberSampler; nsamples = nv(gs.gl.ising.g), 
        showprogress=false)
    prog = Progress(nsamples, dt=showprogress ? 0.1 : Inf, 
        desc="Sampling from Glauber")
    sizehint!(gs.X, nsamples)
    for _ in 1:nsamples
        x = _sample(gs.gl)
        push!(gs.X, x)
        next!(prog)
    end
    return gs
end
function sample(gl::Glauber; kw...) 
    sample!(GlauberSampler(gl); kw...)
end

# return a (T+1) by N matrix
function magnetizations(gs::GlauberSampler) 
    m = mean(gs.X)
    s = std(gs.X) ./ sqrt(length(gs.X))
    return potts2spin.( m .± s )
end

struct SoftMarginSampler{B<:MPdBP, F<:AbstractFloat}
    bp :: B
    X  :: Vector{Matrix{Int}}
    w  :: Vector{F}

    function SoftMarginSampler(bp::MPdBP{q,T,F,U}, 
            X::Vector{Matrix{Int}}, w::Vector{F}) where{q,T,F,U}

        N = nv(bp.g)
        @assert length(X) == length(w)
        @assert all(≥(0), w)
        @assert all(x -> size(x) == (N, T+1), X)

        new{MPdBP{q,T,F,U}, F}(bp, X, w)
    end
end

# a sample with its weight
function onesample!(x::Matrix{Int}, bp::MPdBP{q,T,F,U}) where {q,T,F,U}
    @unpack g, w, ϕ, ψ, p⁰, μ = bp
    N = nv(bp.g)
    @assert size(x) == (N , T+1)
    wg = 1.0

    for i in 1:N
        x[i, 1] = sample_noalloc(p⁰[i])
    end

    for t in 1:T
        for i in 1:N
            ∂i = neighbors(bp.g, i)
            p = [w[i][t](xx, x[∂i, t], x[i, t]) for xx in 1:q]
            xᵢᵗ = sample_noalloc(p)
            x[i, t+1] = xᵢᵗ
            wg *= ϕ[i][t][xᵢᵗ]
        end
    end
    for t in 1:T
        for (i, j, ij) in edges(bp.g)
            wg *= sqrt( ψ[ij][t][x[i,t+1], x[j,t+1]] )
        end
    end
    return x, wg
end
function onesample(bp::MPdBP{q,T,F,U}) where {q,T,F,U}  
    N = nv(bp.g)
    x = zeros(Int, N, T+1)
    onesample!(x, bp)
end

function sample(bp::MPdBP, nsamples::Integer)
    prog = Progress(nsamples, desc="SoftMargin sampling...")
    S = map(1:nsamples) do _
        s = onesample(bp)
        next!(prog)
        s
    end 
    X, w = unzip(S)
    SoftMarginSampler(bp, X, w)
end

# return a (T+1) by N matrix, with errors
function marginals(sms::SoftMarginSampler) 
    @unpack bp, X, w = sms
    N = nv(bp.g); T = getT(bp); q = getq(bp)
    marg = [[zeros(Measurement, q) for t in 0:T] for i in 1:N]
    @assert all(>=(0), w)
    wv = StatsBase.weights(w)
    nsamples = length(X)

    for i in 1:N
        for t in 1:T+1
            x = [xx[i, t] for xx in X]
            mit_avg = StatsBase.proportions(x, q, wv)
            # avoid numerical errors yielding probabilities > 1
            mit_avg = map(x -> x≥1 ? 1 : x, mit_avg)
            mit_var = mit_avg .* (1 .- mit_avg) ./ nsamples
            marg[i][t] .= mit_avg .± sqrt.( mit_var )
        end
    end

   return marg
end

# draw `nobs` observations from the prior
function draw_node_observations!(ϕ::Vector{Vector{Vector{F}}}, 
        X::Matrix{<:Integer}, nobs::Integer; softinf::Real=Inf,
        last_time::Bool=false) where {F<:Real}
    N, T = size(X) .- (0, 1)
    it = if last_time
        sample( collect.(Iterators.product(T:T, 1:N)), nobs)
    else
        sample( collect.(Iterators.product(1:T, 1:N)), nobs)
    end
    softone = logistic(log(softinf)); softzero = logistic(-log(softinf))
    for (t, i) in it
        ϕ[i][t] .= [x==X[i,t+1] ? softone : softzero for x in eachindex(ϕ[i][t])]
    end
    ϕ
end

function draw_node_observations!(bp::MPdBP, nobs::Integer; kw...)
    X, _ = onesample(bp)
    draw_node_observations!(bp.ϕ, X, nobs; kw...)
    nothing
end