import IndexedGraphs: IndexedGraph, outedges, neighbors
import SparseArrays: nzrange
import ProgressMeter: Progress, next!
import Base.Threads: @threads
import LogExpFunctions: logcosh

include("../utils.jl")

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


struct ExactGlauber{T, N, F<:AbstractFloat}
    ising :: Ising{F}
    p⁰ :: Vector{Vector{F}}         # initial state
    ϕ :: Vector{Vector{Vector{F}}}  # observations
    p :: Vector{F}

    function ExactGlauber(ising::Ising{F}, p⁰::Vector{Vector{F}}, 
            ϕ::Vector{Vector{Vector{F}}}) where {F<:AbstractFloat}
        N = length(p⁰)
        @assert length(ϕ) == N
        T = length(ϕ[1])
        @assert all(length(ϕᵢ) == T for ϕᵢ in ϕ)
        p = fill_p(ising, p⁰, ϕ, T, N)
        new{T,N,F}(ising, p⁰, ϕ, p)
    end
end

T = 4
N = 2
p = zeros(2^(N*(T+1)))

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

function fill_p(ising::Ising, p⁰, ϕ, T::Integer, N::Integer; 
        p = ones(2^(N*(T+1))))
    
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


function site_marginals(gl::ExactGlauber{T, N, F}; 
        m = [zeros(fill(2,T+1)...) for i in 1:N]) where {T,N,F}
    prog = Progress(2^(N*(T+1)), desc="Computing site marginals")
    X = zeros(Int, T+1, N)
    for x in 1:2^(N*(T+1))
        X .= int_to_matrix(x-1, (T+1,N))
        for i in 1:N
            m[i][X[:,i]...] += gl.p[x]
        end
        next!(prog)
    end
    @assert all(sum(pᵢ) ≈ 1 for pᵢ in m)
    m
end

function site_time_marginals(gl::ExactGlauber{T, N, F}; 
        m = site_marginals(gl)) where {T,N,F}
    p = [[zeros(2) for t in 0:T] for i in 1:N]
    for i in 1:N
        for t in 1:T+1
            for xᵢᵗ in 1:2
                indices = [s==t ? xᵢᵗ : Colon() for s in 1:T+1]
                p[i][t][xᵢᵗ] = sum(m[i][indices...])
            end
            p[i][t] ./= sum(p[i][t])
        end
    end
    p
end

function site_time_magnetizations(gl::ExactGlauber{T, N, F};
        m = site_marginals(gl), mm = site_time_marginals(gl; m)) where {T,N,F}
    map(1:N) do i
        pᵢ = mm[i]
        reduce.(-, pᵢ)
    end
end