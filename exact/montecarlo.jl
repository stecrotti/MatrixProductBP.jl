import Distributions: sample!
using Measurements
import Statistics: mean, std

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

struct GibbsSampler{T}
    ising :: Ising{T}
    X     :: Vector{Vector{Int}}
    function GibbsSampler(ising::Ising{T}) where {T<:AbstractFloat}
        X = Vector{Int64}[]
        new{T}(ising, X)
    end
end

function sample!(gs::GibbsSampler; nsweeps = nv(gs.ising.g),
        x0 = rand(1:2, nv(gs.ising.g)), nodes=1:nv(ising.g),
        ntherm = 100)
    N = nv(gs.ising.g)
    x = copy(x0)
    for _ in 1:ntherm
        sweep!(x, ising; nodes)
    end
    for _ in 1:nsweeps
        sweep!(x, ising; nodes)
        push!(gs.X, copy(x))
    end
    return gs
end
sample(ising::Ising; kw...) = sample!(GibbsSampler(ising); kw...)

function magnetizations(gs::GibbsSampler) 
    m = potts2spin.( mean(gs.X) )
    s = std(gs.X) / sqrt(length(gs.X))
    m .± s
end
