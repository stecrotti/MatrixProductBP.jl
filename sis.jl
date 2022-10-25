import IndexedGraphs: IndexedGraph

include("dbp_factor.jl")
include("mpdbp.jl")

struct SIS{T, N, F<:AbstractFloat}
    g  :: IndexedGraph
    λ  :: F
    κ  :: F
    p⁰ :: Vector{Vector{F}}          # initial state
    ϕ  :: Vector{Vector{Vector{F}}}  # observations
    function SIS(g::IndexedGraph, λ::F, κ::F, p⁰::Vector{Vector{F}},
            ϕ::Vector{Vector{Vector{F}}}) where {F<:AbstractFloat}
        @assert 0 ≤ λ ≤ 1
        @assert 0 ≤ κ ≤ 1
        N = length(p⁰)
        @assert length(ϕ) == N
        T = length(ϕ[1])
        @assert all(length(ϕᵢ) == T for ϕᵢ in ϕ)
        new{T,N,F}(g, λ, κ, p⁰, ϕ)
    end
end

function mpdbp(sis::SIS{T,N,F}; kw...) where {T,N,F}
    g = IndexedBiDiGraph(sis.g.A)
    w = sis_factors(sis)
    return mpdbp(g, w, q_sis, T, p⁰=sis.p⁰, ϕ=sis.ϕ; kw...)
end

function sis_factors(sis::SIS{T,N,F}) where {T,N,F}
    [fill(SISFactor(sis.λ, sis.κ), T) for i in vertices(sis.g)]
end