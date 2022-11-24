
```
Factor for the factor graph of a model solvable with MPBP.
Any `BPFactor` subtype must implement a functor that computes the Boltzmann
contribution to the joint probability
```
abstract type BPFactor; end


struct MPBP{q,T,F<:Real,U<:BPFactor}
    g  :: IndexedBiDiGraph{Int}          # graph
    w  :: Vector{Vector{U}}              # factors, one per variable
    ϕ  :: Vector{Vector{Vector{F}}}      # vertex-dependent factors
    ψ  :: Vector{Vector{Matrix{F}}}      # edge-dependent factors
    p⁰ :: Vector{Vector{F}}              # prior at time zero
    μ  :: Vector{MPEM2{q,T,F}}           # messages, two per edge
    
    function MPBP(g::IndexedBiDiGraph{Int}, w::Vector{Vector{U}}, 
            ϕ::Vector{Vector{Vector{F}}}, ψ::Vector{Vector{Matrix{F}}},
            p⁰::Vector{Vector{F}}, 
            μ::Vector{MPEM2{q,T,F}}) where {q,T,F<:Real,U<:BPFactor}
    
        @assert length(w) == length(ϕ) == nv(g) "$(length(w)), $(length(ϕ)), $(nv(g))"
        @assert length(ψ) == ne(g)
        @assert all( length(wᵢ) == T for wᵢ in w )
        @assert all( length(ϕ[i][t]) == q for i in eachindex(ϕ) for t in eachindex(ϕ[i]) )
        @assert all( size(ψ[ij][t]) == (q,q) for ij in eachindex(ψ) for t in eachindex(ψ[ij]) )
        @assert check_ψs(ψ, g)
        @assert all( length(pᵢ⁰) == q for pᵢ⁰ in p⁰ )
        @assert all( length(ϕᵢ) == T+1 for ϕᵢ in ϕ )
        @assert all( length(ψᵢ) == T+1 for ψᵢ in ψ )
        @assert length(μ) == ne(g)
        normalize!.(μ)
        return new{q,T,F,U}(g, w, ϕ, ψ, p⁰, μ)
    end
end

getT(::MPBP{q,T,F,U}) where {q,T,F,U} = T
getq(::MPBP{q,T,F,U}) where {q,T,F,U} = q
getN(bp::MPBP) = nv(bp.g)
getU(::MPBP{q,T,F,U}) where {q,T,F,U} = U

# check that observation on edge i→j is the same as the one on j→i
function check_ψs(ψ::Vector{<:Vector{<:Matrix{<:Real}}}, g::IndexedBiDiGraph)
    X = g.X
    N = nv(g)
    rows = rowvals(X)
    vals = nonzeros(X)
    for j in 1:N
        for k in nzrange(X, j)
            i = rows[k]
            if i < j
                ji = k          # idx of edge i→j
                ij = vals[k]    # idx of edge j→i
                check = map(zip(ψ[ij], ψ[ji])) do (ψᵢⱼᵗ, ψⱼᵢᵗ)
                    ψᵢⱼᵗ == ψⱼᵢᵗ'
                end
                all(check) || return false
            end
        end
    end
    return true
end

function mpbp(g::IndexedBiDiGraph{Int}, w::Vector{<:Vector{<:BPFactor}}, 
        q::Int, T::Int; d::Int=1, bondsizes=[1; fill(d, T); 1],
        ϕ = [[ones(q) for t in 0:T] for _ in vertices(g)],
        ψ = [[ones(q,q) for t in 0:T] for _ in edges(g)],
        p⁰ = [ones(q) for i in 1:nv(g)],
        μ = [mpem2(q, T; d, bondsizes) for e in edges(g)])
    return MPBP(g, w, ϕ, ψ, p⁰, μ)
end

function reset_messages!(bp::MPBP)
    for A in bp.μ
        for Aᵗ in A
            Aᵗ .= 1
        end
        normalize!(A)
    end
    nothing
end

function onebpiter!(bp::MPBP{q,T,F,U}, i::Integer; 
    svd_trunc::SVDTrunc=TruncThresh(1e-6)) where {q,T,F,U}

    _onebpiter!(bp, i, f_bp_generic; svd_trunc)
end

function _onebpiter!(bp::MPBP{q,T,F,U}, i::Integer, f_bp::Function; 
        svd_trunc::SVDTrunc=TruncThresh(1e-6)) where {q,T,F,U}
    @unpack g, w, ϕ, ψ, p⁰, μ = bp
    ein = inedges(g,i)
    eout = outedges(g, i)
    A = μ[ein.|>idx]
    @assert all(normalization(a) ≈ 1 for a in A)
    logzᵢ = 0.0
    for (j_ind, e_out) in enumerate(eout)
        B = f_bp(A, p⁰[i], w[i], ϕ[i], ψ[eout.|>idx], j_ind; svd_trunc)
        C = mpem2(B)
        μ[idx(e_out)] = sweep_RtoL!(C; svd_trunc)
        logzᵢ₂ⱼ = normalize!(μ[idx(e_out)])
        logzᵢ += logzᵢ₂ⱼ
    end
    dᵢ = length(ein)
    return (1 / dᵢ) * logzᵢ
end

# A callback to print info and save stuff during the iterations 
struct CB_BP{TP<:ProgressUnknown}
    prog :: TP
    b    :: Vector{Vector{Vector{Float64}}}
    Δs   :: Vector{Float64}
    f    :: Vector{Float64}

    function CB_BP(bp::MPBP{q,T,F,U}; showprogress::Bool=true) where {q,T,F,U}
        @assert q == 2
        dt = showprogress ? 0.1 : Inf
        prog = ProgressUnknown(desc="Running MPBP: iter", dt=dt)
        TP = typeof(prog)
        b = [getindex.(beliefs(bp), 1)] 
        Δs = zeros(0)
        f = zeros(0)
        new{TP}(prog, b, Δs, f)
    end
end

function (cb::CB_BP)(bp::MPBP, it::Integer, logz_msg::Vector)
    bij, logz_belief = pair_beliefs(bp)
    f = bethe_free_energy(bp, logz_msg, logz_belief)
    marg_new = getindex.(beliefs(bp; bij), 1)
    marg_old = cb.b[end]
    Δ = sum(sum(abs, mn .- mo) for (mn, mo) in zip(marg_new, marg_old))
    push!(cb.Δs, Δ)
    push!(cb.f, f)
    push!(cb.b, marg_new)
    next!(cb.prog, showvalues=[(:Δ,Δ)])
    flush(stdout)
    return Δ
end

function iterate!(bp::MPBP; maxiter::Integer=5, 
        svd_trunc::SVDTrunc=TruncThresh(1e-6),
        showprogress=true, cb=CB_BP(bp; showprogress), tol=1e-10, 
        logz_msg = zeros(nv(bp.g)),
        nodes = collect(vertices(bp.g)))
    for it in 1:maxiter
        for i in nodes
            logz_msg[i] = onebpiter!(bp, i; svd_trunc)
        end
        Δ = cb(bp, it, logz_msg)
        Δ < tol && return it, cb
        sample!(nodes, collect(vertices(bp.g)), replace=false)
    end
    return maxiter, cb
end



# compute joint beliefs for all pairs of neighbors
# return also logzᵢⱼ contributions to logzᵢ
function pair_beliefs(bp::MPBP{q,T,F,U}) where {q,T,F,U}
    b = [[zeros(q,q) for _ in 0:T] for _ in 1:(ne(bp.g))]
    # z = ones(nv(bp.g))
    logz = zeros(nv(bp.g))
    X = bp.g.X
    N = nv(bp.g)
    rows = rowvals(X)
    vals = nonzeros(X)
    for j in 1:N
        dⱼ = length(nzrange(X, j))
        for k in nzrange(X, j)
            i = rows[k]
            ji = k          # idx of message i→j
            ij = vals[k]    # idx of message j→i
            μᵢⱼ = bp.μ[ij]; μⱼᵢ = bp.μ[ji]
            bᵢⱼ, zᵢⱼ = pair_belief(μᵢⱼ, μⱼᵢ)
            # z[j] *= zᵢⱼ ^ (1/dⱼ- 1/2)
            logz[j] += (1/dⱼ- 1/2) * log(zᵢⱼ)
            b[ij] .= bᵢⱼ
        end
    end
    # b, z
    b, logz
end

function beliefs(bp::MPBP; bij = pair_beliefs(bp)[1])
    b = map(vertices(bp.g)) do i 
        ij = idx(first(outedges(bp.g, i)))
        bb = bij[ij]
        map(bb) do bᵢⱼᵗ
            bᵢᵗ = vec(sum(bᵢⱼᵗ, dims=2))
        end
    end
    b
end

"""
In this code variables take value in {1,2,...,q} but in models these can correspond to other, more physically significant values (e.g. +1,-1 spins)
This function, if implemented for a subtype of `BPFactor`, converts to the correct values. Those can now be used to compute expectations
By default, nothing happens and the values are just {1,2,...,q}
"""
idx_to_value(x::Integer, ::Type{<:BPFactor}) = x

function marginal_to_expectation(p::Matrix{<:Real}, U::Type{<:BPFactor})
    μ = 0.0
    for xi in axes(p,1) , xj in axes(p, 2)
        μ += idx_to_value(xi, U) * idx_to_value(xj, U) * p[xi, xj]
    end
    μ
end

function marginal_to_expectation(p::Vector{<:Real}, U::Type{<:BPFactor})
    μ = 0.0
    for xi in eachindex(p)
        μ += idx_to_value(xi, U) * p[xi]
    end
    μ
end

function belief_expectations(bp::MPBP{q,T,F,U}; b = beliefs(bp)) where {q,T,F,U}
    map(vertices(bp.g)) do i 
        map(1:T+1) do t
            marginal_to_expectation(b[i][t], U)
        end
    end
end

# compute joint beliefs for all pairs of neighbors for all pairs of times t,u
# p(xᵢᵗ,xⱼᵗ,xᵢᵘ,xⱼᵘ)
function pair_beliefs_tu(bp::MPBP{q,T,F,U}; showprogress::Bool=true) where {q,T,F,U}
    b = [[zeros(q,q,q,q) for _ in 0:T, _ in 0:T] for _ in 1:(ne(bp.g))]
    X = bp.g.X
    N = nv(bp.g)
    rows = rowvals(X)
    vals = nonzeros(X)
    dt = showprogress ? 0.1 : Inf
    prog = Progress(N, desc="Computing pair beliefs at pairs of times"; dt)
    for j in 1:N
        dⱼ = length(nzrange(X, j))
        for k in nzrange(X, j)
            i = rows[k]
            ji = k          # idx of message i→j
            ij = vals[k]    # idx of message j→i
            μᵢⱼ = bp.μ[ij]; μⱼᵢ = bp.μ[ji]
            b[ij] = pair_belief_tu(μᵢⱼ, μⱼᵢ)
        end
        next!(prog)
    end
    b
end

function beliefs_tu(bp::MPBP{q,T,F,U}; 
        bij_tu::Vector{Matrix{Array{F, 4}}} = pair_beliefs_tu(bp)) where {q,T,F,U}
    b = map(vertices(bp.g)) do i 
        ij = idx(first(outedges(bp.g, i)))::Int
        bb = bij_tu[ij]
        map(bb) do bᵢⱼᵗᵘ
            sum(sum(bᵢⱼᵗᵘ, dims=2),dims=4)[:,1,:,1]
        end
    end
    b
end

function autocorrelation(b_tu::Matrix{Matrix{F}}, 
        U::Type{<:BPFactor}) where {F<:Real}
    r = zeros(size(b_tu)...)
    for t in axes(b_tu, 1), u in axes(b_tu, 2)
        r[t,u] = marginal_to_expectation(b_tu[t,u], U)
    end
    r
end

function autocorrelations(bp::MPBP{q,T,F,U}; 
        b_tu = beliefs_tu(bp)) where {q,T,F,U}
    autocorrelations(b_tu, U)
end

function autocorrelations(b_tu::Vector{Matrix{Matrix{F}}},
        U::Type{<:BPFactor}) where {F<:Real}
    [autocorrelation(bi_tu, U) for bi_tu in b_tu]
end

function covariance(r::Matrix{F}, μ::Vector{F}) where {F<:Real}
    c = zero(r)
    for u in axes(r,2), t in 1:u-1
        c[t, u] = r[t, u] - μ[t]*μ[u]
    end
    c
end

# r: autocorrelations, μ: expectations of marginals
function _autocovariances(r::Vector{Matrix{F}}, μ::Vector{Vector{F}}) where {F<:Real}
    map(eachindex(r)) do i
        covariance(r[i], μ[i])
    end
end

function autocovariances(bp::MPBP; r = autocorrelations(bp), 
    m = beliefs(bp))
    U = getU(bp)
    μ = [marginal_to_expectation.(mᵢ, U) for mᵢ in m] 
    _autocovariances(r, μ)
end


function bethe_free_energy(::MPBP, logz_factors, logz_edges)
    - sum(logz_factors) - sum(logz_edges)
end

function bethe_free_energy(bp::MPBP; svd_trunc=TruncThresh(1e-4))
    fa = zeros(getN(bp))
    for i in eachindex(fa)
        logzi = onebpiter!(bp, i; svd_trunc)
        fa[i] -= logzi
    end
    _, logz_edges = pair_beliefs(bp)
    fa .-= logz_edges
    sum(fa)
end

# compute log of prior probability and log of likelihood for a trajectory `x`
function logprior_loglikelihood(bp::MPBP{q,T,F,U}, x::Matrix{<:Integer}) where {q,T,F,U}
    @unpack g, w, ϕ, ψ, p⁰, μ = bp
    N = nv(bp.g)
    @assert size(x) == (N , T+1)
    logp = 0.0; logl = 0.0

    for i in 1:N
        logp += log(p⁰[i][x[i,1]])
        logl += log(ϕ[i][1][x[i,1]])
    end

    for t in 1:T
        for i in 1:N
            ∂i = neighbors(bp.g, i)
            @views logp += log( w[i][t](x[i, t+1], x[∂i, t], x[i, t]) )
            logl += log( ϕ[i][t+1][x[i, t+1]] )
        end
    end
    for t in 1:T+1
        for (i, j, ij) in edges(bp.g)
            logl += 1/2 * log( ψ[ij][t][x[i,t], x[j,t]] )
        end
    end
    return logp, logl
end


#### OLD
# function belief_slow(bp::MPBP, i::Integer; svd_trunc::SVDTrunc=TruncThresh(1e-6))
#     @unpack g, w, ϕ, p⁰, μ = bp
#     A = μ[inedges(g,i).|>idx]
#     B = f_bp(A, p⁰[i], w[i], ϕ[i])
#     C = mpem2(B)
#     sweep_RtoL!(C; svd_trunc)
#     return firstvar_marginals(C)
# end

# function beliefs_slow(bp::MPBP; kw...)
#     [belief_slow(bp, i; kw...) for i in vertices(bp.g)]
# end

# function magnetizations_slow(bp::MPBP{q,T,F,U}; 
#         svd_trunc::SVDTrunc=TruncThresh(1e-6)) where {q,T,F,U}
#     @assert q == 2
#     map(vertices(bp.g)) do i
#         bᵢ = belief(bp, i; svd_trunc)
#         reduce.(-, bᵢ)
#     end
# end
