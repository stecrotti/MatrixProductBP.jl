import Distributions: sample!, sample
using Measurements
import Statistics: mean, std
import UnPack: @unpack
import Distributions: sample, Bernoulli
import Base.Threads: @threads

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