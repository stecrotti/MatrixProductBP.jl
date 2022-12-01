# maps (1,2) -> (1,-1)
potts2spin(x) = 3-2x
spin2potts(σ) = (3+σ)/2

# Ising model with xᵢ ∈ {1,2} mapped onto spins {+1,-1}
struct Ising{F<:AbstractFloat}
    g :: IndexedGraph{Int}
    J :: Vector{F}
    h :: Vector{F}
    β :: F
end

function Ising(J::AbstractMatrix{F}, h::Vector{F}, β::F) where {F<:AbstractFloat}
    Jvec = [J[i,j] for j in axes(J,2) for i in axes(J,1) if i < j && J[i,j]!=0]
    g = IndexedGraph(J)
    Ising(g, Jvec, h, β)
end

function Ising(g::IndexedGraph; J = ones(ne(g)), h = zeros(nv(g)), β = 1.0)
    Ising(g, J, h, β)
end

is_homogeneous(ising::Ising) = all(isequal(ising.J[1]), ising.J)

function local_w(g::IndexedGraph, J::Vector, h::Vector, i::Integer, xᵢ::Integer, 
        xₙᵢ::Vector{<:Integer}, β::Real)
    ei = outedges(g, i)
    isempty(ei) && return exp( β * potts2spin(xᵢ) * h[i] ) / (2cosh( β * potts2spin(xᵢ) * h[i] ))
    ∂i = idx.(ei)
    @assert length(∂i) == length(xₙᵢ)
    Js = @view J[∂i]
    hⱼᵢ = sum( Jᵢⱼ * potts2spin(xⱼ) for (xⱼ, Jᵢⱼ) in zip(xₙᵢ, Js); init=0.0)
    p = exp( β * potts2spin(xᵢ) * (hⱼᵢ + h[i]) ) / (2cosh(β* (hⱼᵢ + h[i])))
    @assert 0 < p <1
    p
end


struct Glauber{T, N, F<:AbstractFloat}
    ising :: Ising{F}
    p⁰ :: Vector{Vector{F}}          # initial state
    ϕ  :: Vector{Vector{Vector{F}}}  # observations
    ψ  :: Vector{Vector{Matrix{F}}}  # edge-dependent factors

    function Glauber(ising::Ising{F}, p⁰::Vector{Vector{F}}, 
            ϕ::Vector{Vector{Vector{F}}}, 
            ψ::Vector{Vector{Matrix{F}}}) where {F<:AbstractFloat}
        N = length(p⁰)
        @assert length(ϕ) == N
        @assert length(ψ) == ne(ising.g)
        T = length(ϕ[1]) - 1
        @assert all(length(ϕᵢ) == T+1 for ϕᵢ in ϕ)
        @assert all(length(ψᵢⱼ) == T+1 for ψᵢⱼ in ψ)
        new{T,N,F}(ising, p⁰, ϕ, ψ)
    end
end

function Glauber(ising::Ising, T::Integer;
        p⁰ = [ones(2) for i in 1:nv(ising.g)],
        ϕ = [[ones(2) for t in 1:T+1] for _ in vertices(ising.g)],
        ψ = [[ones(2,2) for t in 1:T+1] for _ in edges(ising.g)])
   Glauber(ising, p⁰, ϕ, ψ) 
end

### OLD

# # return true if all the ϕ's are uniform, i.e. the dynamics is free
# function is_free_dynamics(gl::Glauber)
#     is_free_nodes = map(ψ) do ϕᵢ
#         map(ϕᵢ) do ϕᵢᵗ
#             all(y->y==ϕᵢᵗ[1], ϕᵢᵗ)
#         end |> all
#     end |> all
#     is_free_edges = map(ψ) do ψᵢⱼ
#         map(ψᵢⱼ) do ψᵢⱼᵗ
#             all(isequal(ψᵢⱼᵗ[1]), ψᵢⱼᵗ)
#         end |> all
#     end |> all
#     return is_free_nodes && is_free_edges
# end

# function exact_prob(gl::Glauber{T,N,F}) where {T,N,F}
#     T*N > 15 && @warn "T*N=$(T*N). This will take some time!"
#     @unpack ising, p⁰, ϕ, ψ = gl
#     p = ones(2^(N*(T+1)))
#     prog = Progress(2^(N*(T+1)), desc="Computing joint probability")
#     X = zeros(Int, T+1, N)
#     for x in 1:2^(N*(T+1))
#         X .= _int_to_matrix(x-1, (T+1,N))
#         for i in 1:N
#             p[x] *= p⁰[i][X[1,i]]
#             ∂i = neighbors(ising.g, i)
#             for t in 1:T
#                 p[x] *= local_w(ising.g, ising.J, ising.h, i, X[t+1,i], 
#                     X[t,∂i], ising.β)
#                 p[x] *= ϕ[i][t][X[t+1,i]]
#             end
#         end
#         for (i, j, ij) in edges(ising.g)
#             # here g is directed!
#             for t in 1:T
#                 p[x] *= ψ[ij][t][X[t+1,i],X[t+1,j]]
#             end
#         end
#         next!(prog)
#     end
#     Z = sum(p)
#     p ./= Z
#     p, Z
# end