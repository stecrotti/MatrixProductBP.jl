"""
`InfiniteRegularGraph` is a data type for an infinite random regular graph, where by
    symmetry all messages and beliefs are equal, therefore it is enough to store one message
    and one belief.
BP functions can be called on an `InfiniteRegularGraph` the same way one would for a normal 
    graph.    
"""
struct InfiniteRegularGraph <: AbstractIndexedDiGraph{Int}
    k :: Int
end

IndexedGraphs.edges(g::InfiniteRegularGraph) = (IndexedEdge(1,1,1) for _ in 1:g.k)
IndexedGraphs.vertices(::InfiniteRegularGraph) = 1:1
IndexedGraphs.ne(g::InfiniteRegularGraph) = 1
IndexedGraphs.nv(::InfiniteRegularGraph) = 1
IndexedGraphs.inedges(g::InfiniteRegularGraph, i::Integer) = ( @assert i == 1; edges(g) )
IndexedGraphs.outedges(g::InfiniteRegularGraph, i::Integer) = inedges(g, i)
IndexedGraphs.issymmetric(::InfiniteRegularGraph) = true
Base.show(io::IO, g::InfiniteRegularGraph) = println(io, "Infinite regular graph of degree ", g.k)
check_ψs(ψ::Vector{<:Vector{<:Matrix{<:Real}}}, g::InfiniteRegularGraph) = true

function mpbp_infinite_graph(k::Integer, wᵢ::Vector{U}, qi::Int,
    ϕᵢ = fill(ones(qi), length(wᵢ));
    ψₖᵢ = fill(ones(qi, qi), length(wᵢ)),
    d::Int=1, bondsizes=[1; fill(d, length(wᵢ)-1); 1], elem_type::Type{F}=Float64) where {U<:BPFactor, F<:Number}

    T = length(wᵢ) - 1
    @assert length(ϕᵢ) == T + 1
    @assert length(ψₖᵢ) == T + 1
    
    g = InfiniteRegularGraph(k)
    μ = flat_mpem2(elem_type, qi, qi, T; d, bondsizes)
    b = flat_mpem1(elem_type, qi, T; d, bondsizes)
    MPBP(g, [wᵢ], [ϕᵢ], [ψₖᵢ], [μ], [b], [zero(elem_type)])
end

function _pair_beliefs!(b, f, bp::MPBP{G,F}) where {G<:InfiniteRegularGraph,F}
    μᵢⱼ = μⱼᵢ = only(bp.μ)
    bᵢⱼ, zᵢⱼ = f(μᵢⱼ, μⱼᵢ, only(bp.ψ))
    logz = [(1/(bp.g.k-1)- 1/2) * log(zᵢⱼ)]
    b[1] = bᵢⱼ
    b, logz
end

function periodic_mpbp_infinite_graph(k::Integer, wᵢ::Vector{U}, qi::Int,
    ϕᵢ = fill(ones(qi), length(wᵢ));
    ψₖᵢ = fill(ones(qi, qi), length(wᵢ)),
    d::Int=1, bondsizes=fill(d, length(wᵢ))) where {U<:BPFactor}

    T = length(wᵢ) - 1
    @assert length(ϕᵢ) == T + 1
    @assert length(ψₖᵢ) == T + 1
    
    g = InfiniteRegularGraph(k)
    μ = rand_periodic_mpem2(qi, qi, T; d, bondsizes)
    b = rand_periodic_mpem1(qi, T; d, bondsizes)
    MPBP(g, [wᵢ], [ϕᵢ], [ψₖᵢ], [μ], [b], [0.0])
end


"""
`InfiniteBipartiteRegularGraph` is a data type for an infinite bipartite random regular graph, where by
    symmetry there are only two types of messages and beliefs, therefore it is enough to store two message
    and two beliefs.
BP functions can be called on an `InfiniteBipartiteRegularGraph` the same way one would for a normal 
    graph.    
"""
struct InfiniteBipartiteRegularGraph <: AbstractIndexedDiGraph{Int}
    k :: NTuple{2, Int}
end

function IndexedGraphs.edges(g::InfiniteBipartiteRegularGraph) 
    return Iterators.flatten(inedges(g, i) for i in 1:2)    
end
IndexedGraphs.vertices(::InfiniteBipartiteRegularGraph) = 1:2
IndexedGraphs.ne(g::InfiniteBipartiteRegularGraph) = 2
IndexedGraphs.nv(::InfiniteBipartiteRegularGraph) = 2
function IndexedGraphs.inedges(g::InfiniteBipartiteRegularGraph, i::Integer)
    @assert i ∈ (1, 2)
    return (IndexedEdge(3-i, i, i) for _ in 1:g.k[i]) 
end
function IndexedGraphs.outedges(g::InfiniteBipartiteRegularGraph, i::Integer)
    @assert i ∈ (1, 2)
    return (IndexedEdge(i, 3-i, 3-i) for _ in 1:g.k[i]) 
end
IndexedGraphs.issymmetric(::InfiniteBipartiteRegularGraph) = true
Base.show(io::IO, g::InfiniteBipartiteRegularGraph) = println(io, "Infinite bipartite regular graph of degrees ", g.k)
check_ψs(ψ::Vector{<:Vector{<:Matrix{<:Real}}}, g::InfiniteBipartiteRegularGraph) = true

function mpbp_infinite_bipartite_graph(k::NTuple{2,Int}, wᵢ::Vector{Vector{U}},
        qi::NTuple{2,Int},
        ϕᵢ = [fill(ones(qi[i]), length(wᵢ[1])) for i in 1:2];
        ψₖᵢ = [fill(ones(qi[i], qi[3-i]), length(wᵢ[1])) for i in 1:2],
        d=(1, 1), bondsizes=ntuple(i->[1; fill(d[i], length(wᵢ[1])-1); 1], 2)) where {U<:BPFactor}

    T = length(wᵢ[1]) - 1
    @assert length(wᵢ[2]) == T + 1
    @assert all(isequal(T+1), length.(ϕᵢ))
    @assert all(isequal(T+1), length.(ψₖᵢ))
    
    g = InfiniteBipartiteRegularGraph(k)
    μ = [flat_mpem2(qi[i], qi[3-i], T; d=d[i], bondsizes=bondsizes[i]) for i in 1:2]
    b = [flat_mpem1(qi[i], T; d=d[i], bondsizes=bondsizes[i]) for i in 1:2]
    MPBP(g, wᵢ, ϕᵢ, ψₖᵢ, μ, b, zeros(2))
end

function _pair_beliefs!(b, f, bp::MPBP{G,F}) where {G<:InfiniteBipartiteRegularGraph,F}
    @assert bp.ψ[1] == bp.ψ[2]
    logz = zeros(2)
    for i in 1:2
        b[i], zᵢⱼ = f(bp.μ[i], bp.μ[3-i], bp.ψ[i])
        logz[i] = (1/(bp.g.k[i]-1)- 1/2) * log(zᵢⱼ)
    end 
    b, logz
end

# the bethe free energy contributions must be reweighted according to the fraction of nodes in each block
function bethe_free_energy(bp::MPBP{<:InfiniteBipartiteRegularGraph})
    k = bp.g.k
    f = bp.f
    return (f[1]*k[2] + f[2]*k[1]) / sum(k)
end