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

edges(g::InfiniteRegularGraph) = (IndexedEdge(1,1,1) for _ in 1:g.k)
vertices(::InfiniteRegularGraph) = 1:1
ne(g::InfiniteRegularGraph) = 1
nv(::InfiniteRegularGraph) = 1
inedges(g::InfiniteRegularGraph, i::Integer) = ( @assert i == 1; edges(g) )
outedges(g::InfiniteRegularGraph, i::Integer) = inedges(g, i)
Base.show(io::IO, g::InfiniteRegularGraph) = println(io, "Infinite regular graph of degree ", g.k)
check_ψs(ψ::Vector{<:Vector{<:Matrix{<:Real}}}, g::InfiniteRegularGraph) = true

function mpbp_infinite_graph(k::Integer, wᵢ::Vector{U}, qi::Int,
    ϕᵢ = fill(ones(qi, qi));
    ψₖᵢ = fill(ones(qi, qi), length(wᵢ)),
    d::Int=1, bondsizes=[1; fill(d, length(wᵢ)-1); 1]) where {U<:BPFactor}

    T = length(wᵢ) - 1
    @assert length(ϕᵢ) == T + 1
    @assert length(ψₖᵢ) == T + 1
    
    g = InfiniteRegularGraph(k)
    μ = mpem2(qi, qi, T; d, bondsizes)
    b = mpem1(qi, T; d, bondsizes)
    MPBP(g, [wᵢ], [ϕᵢ], [ψₖᵢ], [μ], [b], [0.0])
end

# function pair_beliefs(bp::MPBP{G,F}) where {G<:InfiniteRegularGraph,F}
#     μᵢⱼ = μⱼᵢ = bp.μ[1]
#     bᵢⱼ, zᵢⱼ = pair_belief(μᵢⱼ, μⱼᵢ, only(bp.ψ))
#     logz = [(1/(bp.g.k-1)- 1/2) * log(zᵢⱼ)]
#     [bᵢⱼ], logz
# end

function _pair_beliefs!(b, f, bp::MPBP{G,F}) where {G<:InfiniteRegularGraph,F}
    μᵢⱼ = μⱼᵢ = only(bp.μ)
    bᵢⱼ, zᵢⱼ = f(μᵢⱼ, μⱼᵢ, only(bp.ψ))
    logz = [(1/(bp.g.k-1)- 1/2) * log(zᵢⱼ)]
    b[1] = bᵢⱼ
    b, logz
end