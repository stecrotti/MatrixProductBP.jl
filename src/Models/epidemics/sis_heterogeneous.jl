struct SIS_heterogeneous{T, N, F<:Real}
    g  :: IndexedGraph
    λ  :: Matrix{F}
    ρ  :: Vector{F}
    ϕ  :: Vector{Vector{Vector{F}}}  # site observations
    ψ  :: Vector{Vector{Matrix{F}}}  # edge observations
    function SIS_heterogeneous(g::IndexedGraph, λ::Matrix{F}, ρ::Vector{F},
            ϕ::Vector{Vector{Vector{F}}},
            ψ::Vector{Vector{Matrix{F}}}) where {F<:Real}
        @assert size(λ)[1] == size(λ)[2] == nv(g)
        @assert length(ρ) == nv(g)
        @assert all(0 ≤ λᵢⱼ ≤ 1 for λᵢⱼ in λ)
        @assert all(0 ≤ ρᵢ ≤ 1 for ρᵢ in ρ)
        N = nv(g)
        @assert length(ϕ) == N
        T = length(ϕ[1]) - 1
        @assert all(length(ϕᵢ) == T + 1 for ϕᵢ in ϕ)
        @assert length(ψ) == 2*ne(g)
        @assert all(length(ψᵢⱼ) == T + 1 for ψᵢⱼ in ψ)
        new{T,N,F}(g, λ, ρ, ϕ, ψ)
    end
end

function SIS_heterogeneous(g::IndexedGraph{Int}, λ::Matrix{F}, ρ::Vector{F}, T::Int;
        ψ = [[ones(2,2) for t in 0:T] for _ in 1:2*ne(g)],
        γ = 0.5,
        ϕ = [[t == 0 ? (length(γ)==1 ? [1-γ, γ] : [1-γ[i],γ[i]]) : ones(2) for t in 0:T] for i in vertices(g)]) where {F<:Real}
    return SIS_heterogeneous(g, λ, ρ, ϕ, ψ)
end

function sis_heterogeneous_factors(sis::SIS_heterogeneous{T,N,F}) where {T,N,F}
    [fill(SIS_heterogeneousFactor([sis.λ[x,i] for x in eachindex(sis.λ[:,i]) if x!=i], sis.ρ[i]), T + 1) for i in vertices(sis.g)]
end