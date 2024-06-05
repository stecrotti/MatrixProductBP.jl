struct MPBP{G<:AbstractIndexedDiGraph, F<:Real, V<:AbstractVector{<:BPFactor}, M2<:AbstractMPEM2, M1<:AbstractMPEM1}
    g     :: G                              # graph
    w     :: Vector{V}                      # factors, one per variable
    ϕ     :: Vector{Vector{Vector{F}}}      # vertex-dependent factors
    ψ     :: Vector{Vector{Matrix{F}}}      # edge-dependent factors
    μ     :: AtomicVector{M2}               # messages, two per edge
    b     :: Vector{M1}                     # beliefs in matrix product form
    f     :: Vector{F}                      # free energy contributions
    
    function MPBP(g::G, w::Vector{V}, 
            ϕ::Vector{Vector{Vector{F}}},
            ψ::Vector{Vector{Matrix{F}}},
            μ::Vector{M2},
            b::Vector{M1},
            f::Vector{F}) where {G<:AbstractIndexedDiGraph, F<:Real, 
            V<:AbstractVector{<:BPFactor}, M2<:AbstractMPEM2, M1<:AbstractMPEM1}
    
        @assert issymmetric(g)
        T = length(w[1]) - 1
        @assert length(w) == length(ϕ) == length(b) == length(f) == nv(g) "$(length(w)), $(length(ϕ)), $(nv(g))"
        @assert length(ψ) == ne(g)
        @assert all( length(wᵢ) == T + 1 for wᵢ in w )
        @assert all( length(ϕ[i][t]) == nstates(b[i]) for i in eachindex(ϕ) for t in eachindex(ϕ[i]) )
        @assert all( size(ψ[k][t]) == (nstates(b[i]),nstates(b[j])) for (i,j,k) in edges(g), t in 1:T+1 )
        @assert check_ψs(ψ, g)
        @assert all( length(ϕᵢ) == T + 1 for ϕᵢ in ϕ )
        @assert all( length(ψᵢ) == T + 1 for ψᵢ in ψ )
        @assert all( length(μᵢⱼ) == T + 1 for μᵢⱼ in μ)
        @assert all( length(bᵢ) == T + 1 for bᵢ in b )
        @assert length(μ) == ne(g)
        return new{G,F,V,M2,M1}(g, w, ϕ, ψ, AtomicVector(μ), b, f)
    end
end

getT(bp::MPBP) = length(bp.b[1]) - 1
getN(bp::MPBP) = nv(bp.g)
nstates(bp::MPBP, i) = nstates(bp.b[i])

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
                for (ψᵢⱼᵗ, ψⱼᵢᵗ) in zip(ψ[ij], ψ[ji])
                    ψᵢⱼᵗ == ψⱼᵢᵗ' || return false
                end
            end
        end
    end
    return true
end

function mpbp(g::IndexedBiDiGraph{Int}, w::Vector{<:Vector{<:BPFactor}},
        q::AbstractVector{Int}, T::Int; 
        d::Int=1,
        bondsizes=[1; fill(d, T); 1],
        ϕ = [[ones(q[i]) for t in 0:T] for i in vertices(g)],
        ψ = [[ones(q[i],q[j]) for t in 0:T] for (i,j) in edges(g)],
        μ = [flat_mpem2(q[i],q[j], T; d, bondsizes) for (i,j) in edges(g)],
        b = [flat_mpem1(q[i], T; d, bondsizes) for i in vertices(g)],
        f = zeros(nv(g)))
    return MPBP(g, w, ϕ, ψ, μ, b, f)
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
function reset_beliefs!(bp::MPBP)
    for b in bp.b
        for bᵗ in b
            bᵗ .= 1
        end
    end
    nothing
end
function reset_observations!(bp::MPBP)
    for ϕ in bp.ϕ
        for ϕᵗ in ϕ
            ϕᵗ .= 1
        end
    end
    nothing
end
function reset!(bp::MPBP; messages=true, beliefs=true, observations=false)
    messages && reset_messages!(bp)
    beliefs && reset_beliefs!(bp)
    observations && reset_observations!(bp)
    nothing
end

# dynamics is free if there is no reweighting <-> all ϕ's (but the one at time zero) are constant
function is_free_dynamics(bp::MPBP)
    if is_periodic(bp)
        return all(all(all(isequal(first(ϕᵢᵗ)), ϕᵢᵗ) for ϕᵢᵗ in ϕᵢ) for ϕᵢ in bp.ϕ)
    else
        return all(all(all(isequal(first(ϕᵢᵗ)), ϕᵢᵗ) for ϕᵢᵗ in Iterators.drop(ϕᵢ, 1)) for ϕᵢ in bp.ϕ)
    end
end

is_periodic(bp::MPBP{G,F,V,<:MPEM2,<:MPEM1}) where {G,F,V}  = false
is_periodic(bp::MPBP{G,F,V,<:PeriodicMPEM2,<:PeriodicMPEM1}) where {G,F,V} = true

# compute outgoing messages from node `i`
function onebpiter!(bp::MPBP{G,F,V,MsgType}, i::Integer, ::Type{U}; 
        svd_trunc::SVDTrunc=default_truncator(MsgType), damp=0.0) where {U<:BPFactor,G,F,V,MsgType}
    @unpack g, w, ϕ, ψ, μ = bp
    ein = inedges(g,i)
    eout = outedges(g, i)
    A = μ[ein.|>idx]
    @assert all(float(normalization(a)) ≈ 1 for a in A)
    sumlogzᵢ₂ⱼ = 0.0
    for (j_ind, e_out) in enumerate(eout)
        B, logzᵢ₂ⱼ = f_bp(A, w[i], ϕ[i], ψ[eout.|>idx], j_ind; svd_trunc, periodic=is_periodic(bp))
        sumlogzᵢ₂ⱼ += logzᵢ₂ⱼ
        C = mpem2(B)
        μj = compress!(C; svd_trunc, is_orthogonal=:left)
        sumlogzᵢ₂ⱼ += normalize!(μj)
        μ[idx(e_out)] = μj
    end
    dᵢ = length(ein)
    bp.b[i] = onebpiter_dummy_neighbor(bp, i; svd_trunc) |> marginalize
    logzᵢ = real(log(normalization(bp.b[i])))
    bp.f[i] = (dᵢ/2-1)*logzᵢ - (1/2)*sumlogzᵢ₂ⱼ
    nothing
end

function onebpiter!(bp::MPBP{G,F,V,MsgType}, i::Integer; svd_trunc::SVDTrunc=default_truncator(MsgType)) where {G,F,V,MsgType}
    onebpiter!(bp, i, eltype(bp.w[i]); svd_trunc, damp=0.0)
end


function onebpiter_dummy_neighbor(bp::MPBP{G,F,V,MsgType}, i::Integer; 
        svd_trunc::SVDTrunc=default_truncator(MsgType)) where {G,F,V,MsgType}
    @unpack g, w, ϕ, ψ, μ = bp
    ein = inedges(g,i)
    eout = outedges(g, i)
    A = μ[ein.|>idx]
    B, _ = f_bp_dummy_neighbor(A, w[i], ϕ[i], ψ[eout.|>idx]; svd_trunc, periodic=is_periodic(bp))
    C = mpem2(B)
    return compress!(C; svd_trunc, is_orthogonal=:left)
end

# A callback to print info and save stuff during the iterations 
struct CB_BP{TP<:ProgressUnknown, F}
    prog :: TP
    m    :: Vector{Vector{Vector{Float64}}} 
    Δs   :: Vector{Float64}
    f    :: F

    function CB_BP(bp::MPBP; showprogress::Bool=true, f::F=(x,i)->x, info="") where F
        dt = showprogress ? 0.1 : Inf
        isempty(info) || (info *= "\n")
        prog = ProgressUnknown(desc=info*"Running MPBP: iter", dt=dt)
        TP = typeof(prog)
        m = [means(f, bp)]
        Δs = zeros(0)
        new{TP,F}(prog, m, Δs, f)
    end
end

function (cb::CB_BP)(bp::MPBP, it::Integer, svd_trunc::SVDTrunc)
    marg_new = means(cb.f, bp)
    marg_old = cb.m[end]
    Δ = isempty(marg_new) ? NaN : maximum(maximum(abs, mn .- mo) for (mn, mo) in zip(marg_new, marg_old))
    push!(cb.Δs, Δ)
    push!(cb.m, marg_new)
    next!(cb.prog, showvalues=[(:Δ,Δ), summary_compact(svd_trunc)])
    flush(stdout)
    return Δ
end

function iterate!(bp::MPBP{G,F,V,MsgType}; maxiter::Integer=5, 
        svd_trunc::SVDTrunc=default_truncator(MsgType),
        showprogress=true, cb=CB_BP(bp; showprogress), tol=1e-10, 
        nodes = collect(vertices(bp.g)), shuffle_nodes::Bool=true, damp=0.0) where {G,F,V,MsgType}
    for it in 1:maxiter
        Threads.@threads for i in nodes
            onebpiter!(bp, i, eltype(bp.w[i]); svd_trunc, damp)
        end
        Δ = cb(bp, it, svd_trunc)
        Δ < tol && return it, cb
        shuffle_nodes && sample!(nodes, collect(vertices(bp.g)), replace=false)
    end
    return maxiter, cb
end

# compute joint beliefs for all pairs of neighbors
# return also logzᵢⱼ contributions to logzᵢ
function pair_beliefs(bp::MPBP{G,F}) where {G,F}
    b = [[zeros(nstates(bp,i),nstates(bp,j)) for _ in 0:getT(bp)] for (i,j) in edges(bp.g)]
    return _pair_beliefs!(b, pair_belief, bp)
end

# return pair beliefs in MPEM form
function pair_beliefs_as_mpem(bp::MPBP{G,F,V,M2}) where {G,F,V,M2}
    # b = [flat_mpem2(nstates(bp,i),nstates(bp,j), getT(bp)) for (i,j) in edges(bp.g)]
    b = Vector{M2}(undef, ne(bp.g))
    function f(A, B, ψ) 
        C = pair_belief_as_mpem(A, B, ψ)
        C, normalization(C)
    end
    return _pair_beliefs!(b, f, bp)
end

function _pair_beliefs!(b, f, bp::MPBP{G,F}) where {G,F}
    logz = zeros(nv(bp.g))
    X = bp.g.X
    N = nv(bp.g)
    vals = nonzeros(X)
    for j in 1:N
        dⱼ = length(nzrange(X, j))
        for k in nzrange(X, j)
            ij = k          # idx of message i→j
            ji = vals[k]    # idx of message j→i
            μᵢⱼ = bp.μ[ij]; μⱼᵢ = bp.μ[ji]
            bᵢⱼ, zᵢⱼ = f(μᵢⱼ, μⱼᵢ, bp.ψ[ij])
            logz[j] += (1/dⱼ- 1/2) * log(zᵢⱼ)
            b[ij] = bᵢⱼ
        end
    end
    b, logz
end

beliefs(bp::MPBP{G,F}) where {G,F} = marginals.(bp.b)

beliefs_tu(bp::MPBP{G,F}) where {G,F} = twovar_marginals.(bp.b)

expectation(f, p::AbstractMatrix{<:Number}) = sum(f(xi) * f(xj) * p[xi, xj] for xi in axes(p,1), xj in axes(p,2); init=0.0)

expectation(f, p::AbstractVector{<:Number}) = sum(f(xi) * p[xi] for xi in eachindex(p); init=0.0)

function autocorrelations(f, bp::MPBP; showprogress::Bool=false, sites=vertices(bp.g),
        maxdist = getT(bp))
    dt = showprogress ? 0.1 : Inf
    prog = Progress(nv(bp.g); dt, desc="Computing autocorrelations")
    map(sites) do i
        next!(prog)
        expectation.(x->f(x, i), twovar_marginals(bp.b[i]; maxdist))
    end
end

autocorrelations(bp::MPBP; kw...) = autocorrelations((x,i)->x, bp; kw...)

function means(f, bp::MPBP{G,F,V,M2}; sites=vertices(bp.g)) where {G,F,V,M2}
    map(sites) do i
        expectation.(x->f(x, i), marginals(bp.b[i]))
    end
end

# return <f(xᵢᵗ)f(xⱼᵗ)> per each directed edge i->j
function pair_correlations(f, bp::MPBP{G,F,V,M2}) where {G,F,V,M2}
    am = pair_beliefs(bp)[1]
    return [expectation.(f, amij) for amij in am]
end

# return p(xᵢᵗ,xⱼᵗ⁺¹) per each directed edge i->j
function alternate_marginals(bp::MPBP{G,F,V,M2}) where {G,F,V,M2}
    pbs = pair_beliefs_as_mpem(bp)[1]
    tvs = twovar_marginals.(pbs)

    return map(tvs) do tv
        map(1:size(tv,1)-1) do t
            tvt = tv[t,t+1]
            dropdims(sum(tvt; dims=(2,3)); dims=(2,3))
        end
    end
end

# return <f(xᵢᵗ)f(xⱼᵗ⁺¹)> per each directed edge i->j
function alternate_correlations(f, bp::MPBP{G,F,V,M2}) where {G,F,V,M2}
    am = alternate_marginals(bp)
    return [expectation.(f, amij) for amij in am]
end

covariance(r::Matrix{<:Real}, μ::Vector{<:Real}) = r .- μ*μ'

function autocovariances(f, bp::MPBP; sites=vertices(bp.g), kw...)
    μ = means(f, bp; sites)
    r = autocorrelations(f, bp; sites, kw...) 
    covariance.(r, μ)
end

autocovariances(bp::MPBP; kw...) = autocovariances((x,i)->x, bp; kw...)

bethe_free_energy(bp::MPBP) = sum(bp.f)

# compute log of posterior probability for a trajectory `x`
function logprob(bp::MPBP, x::Matrix{<:Integer})
    @unpack g, w, ϕ, ψ, μ = bp
    N = nv(bp.g); T = getT(bp)
    @assert size(x) == (N , T + 1)
    logp = 0.0

    for i in 1:N
        logp += log(ϕ[i][1][x[i,1]])
    end

    for t in 1:T
        for i in 1:N
            ∂i = neighbors(bp.g, i)
            @views logp += log( w[i][t](x[i, t+1], x[∂i, t], x[i, t]) )
            logp += log( ϕ[i][t+1][x[i, t+1]] )
        end
    end
    for t in 1:T+1
        for (i, j, ij) in edges(bp.g)
            logp += 1/2 * log( ψ[ij][t][x[i,t], x[j,t]] )
        end
    end
    return logp
end


# return a vector ψ ready for MPBP starting from observations of the type
#  (i, j, t, ψᵢⱼᵗ)
function pair_observations_directed(O::Vector{<:Tuple{I,I,I,V}}, 
        g::IndexedBiDiGraph{Int}, T::Integer, 
        q::Integer) where {I<:Integer,V<:Matrix{<:Real}}

    @assert all(size(obs[4])==(q,q) for obs in O)
    cnt = 0
    ψ = map(edges(g)) do (i, j, ij)
        map(0:T) do t
            id_ij = findall(obs->obs[1:3]==(i,j,t), O)
            id_ji = findall(obs->obs[1:3]==(j,i,t), O)
            if !isempty(id_ij)
                cnt += 1
                only(O[id_ij])[4]
            elseif !isempty(id_ji)
                cnt += 1
                only(O[id_ji])[4] |> permutedims
            else
                ones(q, q)
            end
        end
    end
    @assert cnt == 2*length(O)
    ψ
end

function pair_observations_nondirected(O::Vector{<:Tuple{I,I,I,V}}, 
        g::IndexedGraph{Int}, T::Integer, 
        q::Integer) where {I<:Integer,V<:Matrix{<:Real}}

    @assert all(size(obs[4])==(q,q) for obs in O)
    cnt = 0
    ψ = map(edges(g)) do (i, j, ij)
        map(0:T) do t
            id = findall(obs->(obs[1:3]==(i,j,t) || obs[1:3]==(j,i,t)), O)
            if !isempty(id)
                cnt += 1
                only(O[id])[4]
            else
                ones(q, q)
            end
        end
    end
    @assert cnt == length(O)
    ψ
end

function pair_obs_undirected_to_directed(ψ_undirected::Vector{<:F}, 
        g::IndexedGraph) where {F<:Vector{<:Matrix}}
    ψ_directed = F[]
    sizehint!(ψ_directed, 2*length(ψ_directed)) 
    A = g.A
    vals = nonzeros(A)
    rows = rowvals(A)

    for j in 1:nv(g)
        for k in nzrange(A, j)
            i = rows[k]
            ij = vals[k]
            if i < j
                push!(ψ_directed, ψ_undirected[ij])
            else
                push!(ψ_directed, [permutedims(ψᵢⱼᵗ) for ψᵢⱼᵗ in ψ_undirected[ij]])
            end
        end
    end

    ψ_directed
end

#### Periodic in time
function periodic_mpbp(g::IndexedBiDiGraph{Int}, w::Vector{<:Vector{<:BPFactor}},
        q::AbstractVector{Int}, T::Int; 
        d::Int=1,
        bondsizes=fill(d, T+1),
        ϕ = [[ones(q[i]) for t in 0:T] for i in vertices(g)],
        ψ = [[ones(q[i],q[j]) for t in 0:T] for (i,j) in edges(g)],
        μ = [flat_periodic_mpem2(q[i],q[j], T; d, bondsizes) for (i,j) in edges(g)],
        b = [flat_periodic_mpem1(q[i], T; d, bondsizes) for i in vertices(g)],
        f = zeros(nv(g)))
    return MPBP(g, w, ϕ, ψ, μ, b, f)
end