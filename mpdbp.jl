include("bp.jl")
include("dbp_factor.jl")

import Graphs: nv, ne, edges, vertices
import IndexedGraphs: IndexedBiDiGraph, inedges, outedges, idx
import UnPack: @unpack
import ProgressMeter: ProgressUnknown, next!
import Random: shuffle!
import Base.Threads: @threads

struct MPdBP{q,T,F<:Real,U<:dBP_Factor}
    g  :: IndexedBiDiGraph{Int}          # graph
    w  :: Vector{Vector{U}}              # factors, one per variable
    ϕ  :: Vector{Vector{Vector{F}}}      # vertex-dependent factors
    p⁰ :: Vector{Vector{F}}              # prior at time zero
    μ  :: Vector{MPEM2{q,T,F}}           # messages, two per edge
    
    function MPdBP(g::IndexedBiDiGraph{Int}, w::Vector{Vector{U}}, 
            ϕ::Vector{Vector{Vector{F}}}, p⁰::Vector{Vector{F}}, 
            μ::Vector{MPEM2{q,T,F}}) where {q,T,F<:Real,U<:dBP_Factor}
    
        @assert length(w) == length(ϕ) == nv(g) "$(length(w)), $(length(ϕ)), $(nv(g))"
        @assert all( length(wᵢ) == T for wᵢ in w )
        @assert all( length(ϕ[i][t]) == q for i in eachindex(ϕ) for t in eachindex(ϕ[i]) )
        @assert all( length(pᵢ⁰) == q for pᵢ⁰ in p⁰ )
        @assert all( length(ϕᵢ) == T for ϕᵢ in ϕ )
        @assert length(μ) == ne(g)
        return new{q,T,F,U}(g, w, ϕ, p⁰, μ)
    end
end

getT(bp::MPdBP{q,T,F,U}) where {q,T,F,U} = T
getq(bp::MPdBP{q,T,F,U}) where {q,T,F,U} = q

function mpdbp(g::IndexedBiDiGraph{Int}, w::Vector{<:Vector{<:dBP_Factor}}, 
        q::Int, T::Int; d::Int=1, bondsizes=[1; fill(d, T); 1],
        ϕ = [[ones(q) for t in 1:T] for i in 1:nv(g)],
        p⁰ = [ones(q) for i in 1:nv(g)],
        μ = [mpem2(q, T; d, bondsizes) for e in edges(g)])
    return MPdBP(g, w, ϕ, p⁰, μ)
end

function onebpiter!(bp::MPdBP, i::Integer; svd_trunc::SVDTrunc=TruncThresh(1e-6))
    @unpack g, w, ϕ, p⁰, μ = bp
    A = μ[inedges(g,i).|>idx]
    for (j_ind, e_out) in enumerate( outedges(g, i) )
        B = f_bp(A, p⁰[i], w[i], ϕ[i], j_ind)
        C = mpem2(B)
        μ[idx(e_out)] = sweep_RtoL!(C; svd_trunc)
        normalize_eachmatrix!(μ[idx(e_out)])
    end
    return nothing
end

struct CB_BP{TP<:ProgressUnknown}
    prog :: TP
    mag :: Vector{Vector{Vector{Float64}}}
    Δs :: Vector{Float64}
    function CB_BP(bp::MPdBP{q,T,F,U}) where {q,T,F,U}
        @assert q == 2
        prog = ProgressUnknown(desc="Running MPdBP: iter")
        TP = typeof(prog)
        mag = [magnetizations(bp)] 
        Δs = zeros(0)
        new{TP}(prog, mag, Δs)
    end
end

function (cb::CB_BP)(bp::MPdBP, it::Integer)
    mag_new = magnetizations(bp)
    mag_old = cb.mag[end]
    Δ = sum(sum(abs, mn .- mo) for (mn,mo) in zip(mag_new,mag_old))
    push!(cb.Δs, Δ)
    next!(cb.prog, showvalues=[(:Δ,Δ)])
    push!(cb.mag, mag_new)
    return Δ
end

function iterate!(bp::MPdBP; maxiter=5, svd_trunc::SVDTrunc=TruncThresh(1e-6),
        cb=CB_BP(bp), tol=1e-10,
        nodes = collect(vertices(bp.g)))
    for it in 1:maxiter
        @threads for i in nodes
            onebpiter!(bp, i; svd_trunc)
        end
        Δ = cb(bp, it)
        Δ < tol && return it, cb
        shuffle!(nodes)
    end
    return maxiter, cb
end

function belief(bp::MPdBP, i::Integer; svd_trunc::SVDTrunc=TruncThresh(1e-6))
    @unpack g, w, ϕ, p⁰, μ = bp
    A = μ[inedges(g,i).|>idx]
    B = f_bp(A, p⁰[i], w[i], ϕ[i])
    C = mpem2(B)
    sweep_RtoL!(C; svd_trunc)
    return firstvar_marginals(C)
end

function beliefs(bp::MPdBP; kw...)
    [belief(bp, i; kw...) for i in vertices(bp.g)]
end

function magnetizations(bp::MPdBP{q,T,F,U}; 
        svd_trunc::SVDTrunc=TruncThresh(1e-6)) where {q,T,F,U}
    @assert q == 2
    map(vertices(bp.g)) do i
        bᵢ = belief(bp, i; svd_trunc)
        reduce.(-, bᵢ)
    end
end

