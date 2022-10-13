include("bp.jl")
include("dbp_factor.jl")
include("exact/exact_glauber.jl")

import Graphs: nv, ne, edges, vertices
import IndexedGraphs: IndexedBiDiGraph, inedges, outedges, idx
import UnPack: @unpack
import ProgressMeter: ProgressUnknown, next!

struct MPdBP{q,T,F<:Real,U<:dBP_Factor}
    g  :: IndexedBiDiGraph{Int}          # graph
    w  :: Vector{Vector{U}}              # factors, one per variable
    ϕ  :: Vector{Vector{Vector{F}}}      # vertex-dependent factors
    p⁰ :: Vector{Vector{F}}              # prior at time zero
    μ  :: Vector{MPEM2{q,T,F}}           # messages, two per edge
    
    function MPdBP(g::IndexedBiDiGraph{Int}, w::Vector{Vector{U}}, 
            ϕ::Vector{Vector{Vector{F}}}, p⁰::Vector{Vector{F}}, 
            μ::Vector{MPEM2{q,T,F}}) where {q,T,F<:Real,U<:dBP_Factor}
    
        @assert length(w) == length(ϕ) == nv(g)
        @assert all( length(wᵢ) == T for wᵢ in w )
        @assert all( length(ϕ[i][t]) == q for i in eachindex(ϕ) for t in eachindex(ϕ[i]) )
        @assert all( length(pᵢ⁰) == q for pᵢ⁰ in p⁰ )
        @assert all( length(ϕᵢ) == T for ϕᵢ in ϕ )
        @assert length(μ) == ne(g)
        return new{q,T,F,U}(g, w, ϕ, p⁰, μ)
    end
end

function mpdbp(g::IndexedBiDiGraph{Int}, w::Vector{<:Vector{<:dBP_Factor}}, 
        q::Int, T::Int; d::Int=2, bondsizes=[1; fill(d, T); 1],
        ϕ = [[rand(q) for t in 1:T] for i in 1:nv(g)],
        p⁰ = [rand(q) for i in 1:nv(g)],
        μ = [mpem2(q, T; d, bondsizes) for e in edges(g)])
    return MPdBP(g, w, ϕ, p⁰, μ)
end

function onebpiter!(bp::MPdBP, i::Integer; ε=1e-6)
    @unpack g, w, ϕ, p⁰, μ = bp
    A = μ[inedges(g,i).|>idx]
    for (j_ind, e_out) in enumerate( outedges(g, i) )
        B = f_bp(A, p⁰[i], w[i], ϕ[i], j_ind)
        C = mpem2(B)
        # @show norm( μ[idx(e_out)] -  normalize!(sweep_RtoL!(deepcopy(C); ε)))
        μ[idx(e_out)] = sweep_RtoL!(C; ε)
        normalize!(μ[idx(e_out)], norm_fast_R)
    end
    return nothing
end

struct CB_BP{TP<:ProgressUnknown}
    prog :: TP
    mag :: Vector{Vector{Float64}}
    Δs :: Vector{Float64}
    function CB_BP(bp::MPdBP{q,T,F,U}) where {q,T,F,U}
        @assert q == 2
        prog = ProgressUnknown()
        TP = typeof(prog)
        mag = magnetizations(bp) 
        Δs = zeros(0)
        new{TP}(prog, mag, Δs)
    end
end

function (cb::CB_BP)(bp::MPdBP, it::Integer)
    mag_new = magnetizations(bp)
    mag_old = cb.mag
    Δ = sum(sum(abs, mn .- mo) for (mn,mo) in zip(mag_new,mag_old))
    push!(cb.Δs, Δ)
    next!(cb.prog, showvalues=[(:it,it), (:Δ,Δ)])
    cb.mag .= mag_new
    return nothing
end

function iterate!(bp::MPdBP; maxiter=5, ε=1e-6, cb=CB_BP(bp))
    for it in 1:maxiter
        for i in vertices(bp.g)
            onebpiter!(bp, i; ε)
        end
        cb(bp, it)
    end
    return Δ
end

function belief(bp::MPdBP, i::Integer; ε=1e-6)
    @unpack g, w, ϕ, p⁰, μ = bp
    A = μ[inedges(g,i).|>idx]
    B = f_bp(A, p⁰[i], w[i], ϕ[i])
    C = mpem2(B)
    sweep_RtoL!(C; ε)
    return firstvar_marginals(C)
end

function beliefs(bp::MPdBP; ε=1e-6)
    [belief(bp, i; ε) for i in vertices(bp.g)]
end

function magnetizations(bp::MPdBP{q,T,F,U}; ε=1e-6) where {q,T,F,U}
    @assert q == 2
    map(vertices(bp.g)) do i
        bᵢ = belief(bp, i; ε)
        reduce.(-, bᵢ)
    end
end

function mpdbp(gl::ExactGlauber{T,N,F}) where {T,N,F<:AbstractFloat}
    g = IndexedBiDiGraph(gl.ising.g.A)
    w = glauber_factors(gl)
    ϕ = gl.ϕ
    p⁰ = gl.p⁰
    return mpdbp(g, w, 2, T; ϕ, p⁰)
end