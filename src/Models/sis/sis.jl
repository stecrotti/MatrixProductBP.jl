struct SIS{T, N, F<:Real}
    g  :: IndexedGraph
    λ  :: F
    κ  :: F
    p⁰ :: Vector{Vector{F}}          # initial state
    ϕ  :: Vector{Vector{Vector{F}}}  # site observations
    ψ  :: Vector{Vector{Matrix{F}}}  # edge observations
    function SIS(g::IndexedGraph, λ::F, κ::F, p⁰::Vector{Vector{F}},
            ϕ::Vector{Vector{Vector{F}}},
            ψ::Vector{Vector{Matrix{F}}}) where {F<:Real}
        @assert 0 ≤ λ ≤ 1
        @assert 0 ≤ κ ≤ 1
        N = length(p⁰)
        @assert length(ϕ) == nv(g) == N
        T = length(ϕ[1])
        @assert all(length(ϕᵢ) == T for ϕᵢ in ϕ)
        @assert length(ψ) == 2*ne(g)
        @assert all(length(ψᵢⱼ) == T for ψᵢⱼ in ψ)
        new{T,N,F}(g, λ, κ, p⁰, ϕ, ψ)
    end
end

function SIS(g::IndexedGraph{Int}, λ::F, κ::F, T::Int;
        ϕ = [[ones(2) for t in 1:T] for _ in vertices(g)],
        ψ = [[ones(2,2) for t in 1:T] for _ in 1:2*ne(g)],
        γ = 0.5,
        p⁰ = [[1-γ,γ] for i in 1:nv(g)]) where {F<:Real}
    return SIS(g, λ, κ, p⁰, ϕ, ψ)
end

function sis_factors(sis::SIS{T,N,F}) where {T,N,F}
    [fill(SISFactor(sis.λ, sis.κ), T) for i in vertices(sis.g)]
end