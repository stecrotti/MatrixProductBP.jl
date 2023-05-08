# maps (1,2) -> (1,-1)
potts2spin(x) = 3-2x
spin2potts(σ) = (3-σ)/2

# Ising model with xᵢ ∈ {1,2} mapped onto spins {+1,-1}
struct Ising{F<:AbstractFloat}
    g :: IndexedGraph{Int}
    J :: Vector{F}
    h :: Vector{F}
    β :: F

    function Ising(g::IndexedGraph{Int}, J::Vector{F}, h::Vector{F}, β::F) where {F<:AbstractFloat}
        N = nv(g); E = ne(g)
        @assert length(J) == E
        @assert length(h) == N
        new{F}(g, J, h, β)
    end
end

function Ising(J::AbstractMatrix{F}, h::Vector{F}, β::F) where {F<:AbstractFloat}
    Jvec = [J[i,j] for j in axes(J,2) for i in axes(J,1) if i < j && J[i,j]!=0]
    g = IndexedGraph(Symmetric(J, :L))
    Ising(g, Jvec, h, β)
end

function Ising(g::IndexedGraph; J = ones(ne(g)), h = zeros(nv(g)), β = 1.0)
    Ising(g, J, h, β)
end

is_absJ_const(ising::Ising) = all(J->abs(J)==abs(ising.J[1]), ising.J)
is_homogeneous(ising::Ising) = all(isequal(ising.J[1]), ising.J)
struct Glauber{T, N, F<:AbstractFloat}
    ising :: Ising{F}
    ϕ  :: Vector{Vector{Vector{F}}}  # observations
    ψ  :: Vector{Vector{Matrix{F}}}  # edge-dependent factors

    function Glauber(ising::Ising{F},
            ϕ::Vector{Vector{Vector{F}}}, 
            ψ::Vector{Vector{Matrix{F}}}) where {F<:AbstractFloat}
        N = length(ϕ)
        @assert length(ψ) == ne(ising.g)
        T = length(ϕ[1]) - 1
        @assert all(length(ϕᵢ) == T+1 for ϕᵢ in ϕ)
        @assert all(length(ψᵢⱼ) == T+1 for ψᵢⱼ in ψ)
        new{T,N,F}(ising, ϕ, ψ)
    end
end

function Glauber(ising::Ising, T::Integer;
        ϕ = [[ones(2) for t in 1:T+1] for _ in vertices(ising.g)],
        ψ = [[ones(2,2) for t in 1:T+1] for _ in edges(ising.g)])
   Glauber(ising, ϕ, ψ) 
end