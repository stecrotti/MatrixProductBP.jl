import IndexedGraphs: IndexedGraph, outedges, neighbors, idx
import SparseArrays: nzrange
import ProgressMeter: Progress, next!
import Base.Threads: @threads
import UnPack: @unpack

include("./utils.jl")

# Ising model with xᵢ ∈ {1,2}
struct Ising{F<:AbstractFloat}
    g :: IndexedGraph{Int}
    J :: Vector{F}
    h :: Vector{F}
    β :: F
end

function Ising(J::AbstractMatrix{F}, h::Vector{F}, β::F) where {F<:AbstractFloat}
    gg = SimpleGraph(J)
    Jvec = [J[i,j] for j in axes(J,2) for i in axes(J,1) if i < j && J[i,j]!=0]
    g = IndexedGraph(gg)
    Ising(g, Jvec, h, β)
end

function local_w(g::IndexedGraph, J::Vector, h::Vector, i::Integer, xᵢ::Integer, 
        xₙᵢ::Vector{<:Integer}, β::Real)
    ei = outedges(g, i)
    isempty(ei) && return exp( β * potts2spin(xᵢ) * h[i] )
    ∂i = idx.(ei)
    @assert length(∂i) == length(xₙᵢ)
    Js = @view J[∂i]
    hⱼᵢ = sum( Jᵢⱼ * potts2spin(xⱼ) for (xⱼ, Jᵢⱼ) in zip(xₙᵢ, Js), init=0.0)
    exp( β * potts2spin(xᵢ) * (hⱼᵢ + h[i]) ) / (2cosh(β* (hⱼᵢ + h[i])))
end


struct Glauber{T, N, F<:AbstractFloat}
    ising :: Ising{F}
    p⁰ :: Vector{Vector{F}}         # initial state
    ϕ :: Vector{Vector{Vector{F}}}  # observations

    function Glauber(ising::Ising{F}, p⁰::Vector{Vector{F}}, 
            ϕ::Vector{Vector{Vector{F}}}) where {F<:AbstractFloat}
        N = length(p⁰)
        @assert length(ϕ) == N
        T = length(ϕ[1])
        @assert all(length(ϕᵢ) == T for ϕᵢ in ϕ)
        new{T,N,F}(ising, p⁰, ϕ)
    end
end

function bitarr_to_int(arr::BitArray, s=big(0))
    v = 1
    for i in view(arr,length(arr):-1:1)
        s += v*i
        v <<= 1
    end 
    return s
end

function int_to_matrix(x::Integer, dims)
    y = digits(x, base=2, pad=prod(dims))
    return reshape(y, dims) .+ 1
end

function exact_prob(gl::Glauber{T,N,F}) where {T,N,F}
    T*N > 15 && @warn "T*N=$(T*N). This will take some time!"
    @unpack ising, p⁰, ϕ = gl
    p = ones(2^(N*(T+1)))
    prog = Progress(2^(N*(T+1)), desc="Computing joint probability")
    X = zeros(Int, T+1, N)
    for x in 1:2^(N*(T+1))
        X .= int_to_matrix(x-1, (T+1,N))
        for i in 1:N
            p[x] *= p⁰[i][X[1,i]]
            ∂i = neighbors(ising.g, i)
            for t in 1:T
                p[x] *= local_w(ising.g, ising.J, ising.h, i, X[t+1,i], 
                    X[t,∂i], ising.β)
                p[x] *= ϕ[i][t][X[t+1,i]]
            end
        end
        next!(prog)
    end
    p ./= sum(p)
end


function site_marginals(gl::Glauber{T, N, F}; 
        p = exact_prob(gl),
        m = [zeros(fill(2,T+1)...) for i in 1:N]) where {T,N,F}
    prog = Progress(2^(N*(T+1)), desc="Computing site marginals")
    X = zeros(Int, T+1, N)
    for x in 1:2^(N*(T+1))
        X .= int_to_matrix(x-1, (T+1,N))
        for i in 1:N
            m[i][X[:,i]...] += p[x]
        end
        next!(prog)
    end
    @assert all(sum(pᵢ) ≈ 1 for pᵢ in m)
    m
end

function site_time_marginals(gl::Glauber{T, N, F}; 
        m = site_marginals(gl)) where {T,N,F}
    pp = [[zeros(2) for t in 0:T] for i in 1:N]
    for i in 1:N
        for t in 1:T+1
            for xᵢᵗ in 1:2
                indices = [s==t ? xᵢᵗ : Colon() for s in 1:T+1]
                pp[i][t][xᵢᵗ] = sum(m[i][indices...])
            end
            pp[i][t] ./= sum(pp[i][t])
        end
    end
    pp
end

function site_time_magnetizations(gl::Glauber{T, N, F};
        mm = site_time_marginals(gl)) where {T,N,F}
    map(1:N) do i
        pᵢ = mm[i]
        reduce.(-, pᵢ)
    end
end

# return true if all the ϕ's are uniform, i.e. the dynamics is free
function is_free_dynamics(gl::Glauber)
    map(ϕ) do ϕᵢ
        map(ϕᵢ) do ϕᵢᵗ
            all(y->y==ϕᵢᵗ[1], ϕᵢᵗ)
        end |> all
    end |> all
end