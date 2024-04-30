struct SIRS{T, N, F<:Real}
    g  :: IndexedGraph
    λ  :: F
    ρ  :: F
    σ  :: F
    α  :: F
    ϕ  :: Vector{Vector{Vector{F}}}  # site observations
    ψ  :: Vector{Vector{Matrix{F}}}  # edge observations
    function SIRS(g::IndexedGraph, λ::F, ρ::F, σ::F, α::F,
            ϕ::Vector{Vector{Vector{F}}},
            ψ::Vector{Vector{Matrix{F}}}) where {F<:Real}
        @assert 0 ≤ λ ≤ 1
        @assert 0 ≤ ρ ≤ 1
        @assert 0 ≤ α ≤ 1
        N = nv(g)
        @assert length(ϕ) == N
        T = length(ϕ[1]) - 1
        @assert all(length(ϕᵢ) == T + 1 for ϕᵢ in ϕ)
        @assert length(ψ) == 2*ne(g)
        @assert all(length(ψᵢⱼ) == T + 1 for ψᵢⱼ in ψ)
        new{T,N,F}(g, λ, ρ, σ, α, ϕ, ψ)
    end
end

function SIRS(g::IndexedGraph{Int}, λ::F, ρ::F, σ::F, T::Int;
        ψ = [[ones(3,3) for t in 0:T] for _ in 1:2*ne(g)],
        γ = 0.5, α=0.0,
        ϕ = [[t == 0 ? [1-γ, γ, 0.0] : ones(3) for t in 0:T] for _ in vertices(g)]) where {F<:Real}
    return SIRS(g, λ, ρ, σ, α, ϕ, ψ)
end

function sirs_factors(sirs::SIRS{T,N,F}) where {T,N,F}
    [fill(SIRSFactor(sirs.λ, sirs.ρ, sirs.σ; α=sirs.α), T + 1) for i in vertices(sirs.g)]
end