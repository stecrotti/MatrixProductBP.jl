struct SIS{T, N, F<:Real}
    g  :: IndexedGraph
    λ  :: F
    ρ  :: F
    ϕ  :: Vector{Vector{Vector{F}}}  # site observations
    ψ  :: Vector{Vector{Matrix{F}}}  # edge observations
    function SIS(g::IndexedGraph, λ::F, ρ::F,
            ϕ::Vector{Vector{Vector{F}}},
            ψ::Vector{Vector{Matrix{F}}}) where {F<:Real}
        @assert 0 ≤ λ ≤ 1
        @assert 0 ≤ ρ ≤ 1
        N = nv(g)
        @assert length(ϕ) == N
        T = length(ϕ[1]) - 1
        @assert all(length(ϕᵢ) == T + 1 for ϕᵢ in ϕ)
        @assert length(ψ) == 2*ne(g)
        @assert all(length(ψᵢⱼ) == T + 1 for ψᵢⱼ in ψ)
        new{T,N,F}(g, λ, ρ, ϕ, ψ)
    end
end

function SIS(g::IndexedGraph{Int}, λ::F, ρ::F, T::Int;
        ψ = [[ones(2,2) for t in 0:T] for _ in 1:2*ne(g)],
        γ = 0.5,
        ϕ = [[t == 0 ? [1-γ, γ] : ones(2) for t in 0:T] for _ in vertices(g)]) where {F<:Real}
    return SIS(g, λ, ρ, ϕ, ψ)
end

function sis_factors(sis::SIS{T,N,F}) where {T,N,F}
    [fill(SISFactor(sis.λ, sis.ρ), T + 1) for i in vertices(sis.g)]
end