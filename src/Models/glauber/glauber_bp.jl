const q_glauber = 2

abstract type GlauberFactor <: BPFactor; end

struct GenericGlauberFactor{T<:Real} <: GlauberFactor 
    βJ :: Vector{T}      
    βh :: T
end

struct HomogeneousGlauberFactor{T<:Real} <: GlauberFactor 
    βJ :: Vector{T}      
    βh :: T

    function HomogeneousGlauberFactor(βJ::Vector{T}, βh::T) where {T<:Real}
        all(isequal(βJ[1]), βJ) || throw(ArgumentError("Couplings should be uniform"))
        new{T}(βJ, βh)
    end
end

function HomogeneousGlauberFactor(J::Vector{T}, h::T, β::T) where {T<:Real}
    HomogeneousGlauberFactor(J.*β, h*β)
end

function GenericGlauberFactor(J::Vector{T}, h::T, β::T) where {T<:Real}
    GenericGlauberFactor(J.*β, h*β)
end

function (fᵢ::GlauberFactor)(xᵢᵗ⁺¹::Integer, xₙᵢᵗ::AbstractVector{<:Integer}, 
        xᵢᵗ::Integer)
    @assert xᵢᵗ⁺¹ ∈ 1:q_glauber
    @assert all(x ∈ 1:q_glauber for x in xₙᵢᵗ)
    @assert length(xₙᵢᵗ) == length(fᵢ.βJ)

    hⱼᵢ = sum( Jᵢⱼ * potts2spin(xⱼᵗ) for (xⱼᵗ,Jᵢⱼ) in zip(xₙᵢᵗ, fᵢ.βJ))
    E = - potts2spin(xᵢᵗ⁺¹) * (hⱼᵢ + fᵢ.βh)
    exp( -E ) / (2cosh(E))
end

function mpbp(gl::Glauber{T,N,F}; kw...) where {T,N,F<:AbstractFloat}
    g = IndexedBiDiGraph(gl.ising.g.A)
    w = glauber_factors(gl.ising, T)
    ϕ = gl.ϕ
    ψ = pair_obs_undirected_to_directed(gl.ψ, gl.ising.g)
    p⁰ = gl.p⁰
    return mpbp(g, w, 2, T; ϕ, ψ, p⁰, kw...)
end


# construct an array of GlauberFactors corresponding to gl
function glauber_factors(ising::Ising, T::Integer)
    map(1:nv(ising.g)) do i
        ei = outedges(ising.g, i)
        ∂i = idx.(ei)
        J = ising.J[∂i]
        h = ising.h[i]
        wᵢᵗ = if is_homogeneous(ising)
            HomogeneousGlauberFactor(J, h, ising.β)
        else
            GenericGlauberFactor(J, h, ising.β)
        end
        fill(wᵢᵗ, T)
    end
end

idx_to_value(x::Integer, ::Type{<:GlauberFactor}) = potts2spin(x)

# compute outgoing message efficiently for any degree
# return a `MPMEM3` just like `f_bp`
function f_bp_glauber(A::Vector{MPEM2{q,T,F}}, pᵢ⁰, wᵢ, ϕᵢ, ψₙᵢ, j::Integer;
        svd_trunc=TruncThresh(1e-6)) where {q,T,F}
    d = length(A) - 1   # number of neighbors other than j
    βJ = wᵢ[1].βJ[1]
    @assert all(all(βJij == βJ for βJij in wᵢᵗ.βJ) for wᵢᵗ in wᵢ)
    βh = wᵢ[1].βh
    @assert all(wᵢᵗ.βh  == βh for wᵢᵗ in wᵢ)
    # @assert all(wᵢᵗ.βh == 0 for wᵢᵗ in wᵢ)
    @assert j ∈ eachindex(A)

    # initialize recursion
    M = reshape([1.0 1; 0 0], (1,1,q,q))
    mᵢⱼₗ₁ = MPEM2( fill(M, T+1) )
  
    l = 1
    for k in eachindex(A)
        k == j && continue
        mᵢⱼₗ₁ = f_bp_partial_glauber(A[k], mᵢⱼₗ₁, l)
        l += 1
        # SVD L to R with no truncation
        sweep_LtoR!(mᵢⱼₗ₁, svd_trunc=TruncThresh(0.0))
        # SVD R to L with truncations
        sweep_RtoL!(mᵢⱼₗ₁; svd_trunc)
    end
   
    # combine the last partial message with p(xᵢᵗ⁺¹|xᵢᵗ, xⱼᵗ, yᵗ)
    B = f_bp_partial_ij_glauber(mᵢⱼₗ₁, βJ, βh, pᵢ⁰, ϕᵢ, d)

    return B
end

# the sum of n spins can be one of (n+1) values. We sort them increasingly and
#  index them by k
function _idx_map(n::Integer, k::Integer) 
    @assert n ≥ 0
    @assert k ∈ 1:(n+1)
    return - n + 2*(k-1)
end

# compute message m(i→j, l) from m(i→j, l-1) 
# returns an `MPEM2` [Aᵗᵢⱼ,ₗ(yₗᵗ,xᵢᵗ)]ₘₙ is stored as a 4-array A[m,n,yₗᵗ,xᵢᵗ]
function f_bp_partial_glauber(mₗᵢ::MPEM2{q,T,F}, mᵢⱼₗ₁::MPEM2{q1,T,F}, 
       l::Integer) where {q,q1,T,F}
    @assert q==q_glauber
    AA = Vector{Array{F,4}}(undef, T+1)

    for t in eachindex(AA)
        Aᵗ = kron2(mₗᵢ[t], mᵢⱼₗ₁[t])
        nrows = size(Aᵗ, 1); ncols = size(Aᵗ, 2)
        AAᵗ = zeros(nrows, ncols, l+1, l+1)
        for zₗᵗ in 1:(l+1)    # loop over 1:(l+1) but then y take +/- values
            yₗᵗ = _idx_map(l, zₗᵗ)
            for xᵢᵗ in 1:q
                for zₗ₁ᵗ in 1:l
                    yₗ₁ᵗ = _idx_map(l-1, zₗ₁ᵗ) 
                    for xₗᵗ in 1:q
                        p = prob_partial_msg_glauber(yₗᵗ, yₗ₁ᵗ, xₗᵗ)
                        AAᵗ[:,:,zₗᵗ,xᵢᵗ] .+= p * Aᵗ[:,:,xᵢᵗ,xₗᵗ,zₗ₁ᵗ] 
                    end
                end
            end
        end
        AA[t] = AAᵗ
    end

    return MPEM2(AA)
end

# compute m(i→j) from m(i→j,d)
function f_bp_partial_ij_glauber(A::MPEM2{Q,T,F}, βJ::Real, βh::Real, pᵢ⁰, ϕᵢ, 
        d::Integer) where {Q,T,F}
    q = q_glauber
    B = Vector{Array{F,5}}(undef, T+1)

    A⁰ = A[begin]
    nrows = size(A⁰, 1); ncols = size(A⁰, 2)
    B⁰ = zeros(q, q, nrows, ncols, q)

    for xᵢ¹ in 1:q, xᵢ⁰ in 1:q
        for z⁰ in 1:(d+1)
            y⁰ = _idx_map(d, z⁰)
            for xⱼ⁰ in 1:q
                p = prob_ijy_glauber(xᵢ¹, xⱼ⁰, y⁰, βJ, βh)
                B⁰[xᵢ⁰,xⱼ⁰,1,:,xᵢ¹] .+= p * A⁰[1,:,z⁰,xᵢ⁰]
            end
        end
        B⁰[xᵢ⁰,:,:,:,xᵢ¹] .*= ϕᵢ[1][xᵢ⁰] * ϕᵢ[2][xᵢ¹] * pᵢ⁰[xᵢ⁰] 
    end
    B[begin] = B⁰

    for t in 1:T-1
        Aᵗ = A[begin+t]
        nrows = size(Aᵗ, 1); ncols = size(Aᵗ, 2)
        Bᵗ = zeros(q, q, nrows, ncols, q)

        for xᵢᵗ⁺¹ in 1:q
            for xᵢᵗ in 1:q
                for xⱼᵗ in 1:q
                    for zᵗ in 1:(d+1)
                        yᵗ = _idx_map(d, zᵗ)
                        p = prob_ijy_glauber(xᵢᵗ⁺¹, xⱼᵗ, yᵗ, βJ, βh)
                        Bᵗ[xᵢᵗ,xⱼᵗ,:,:,xᵢᵗ⁺¹] .+= p * Aᵗ[:,:,zᵗ,xᵢᵗ]
                    end
                end
            end
            Bᵗ[:,:,:,:,xᵢᵗ⁺¹] *= ϕᵢ[t+2][xᵢᵗ⁺¹]
        end
        any(isnan, Bᵗ) && println("NaN in tensor at time $t")
        B[begin+t] = Bᵗ
    end

    Aᵀ = A[end]
    nrows = size(Aᵀ, 1); ncols = size(Aᵀ, 2)
    Bᵀ = zeros(q, q, nrows, ncols, q)

    for xᵢᵀ⁺¹ in 1:q
        for xᵢᵀ in 1:q
            for xⱼᵀ in 1:q
                for zᵀ in 1:(d+1)
                    yᵀ = _idx_map(d, zᵀ)
                    Bᵀ[xᵢᵀ,xⱼᵀ,:,:,xᵢᵀ⁺¹] .+= Aᵀ[:,:,zᵀ,xᵢᵀ]
                end
            end
        end
    end
    B[end] = Bᵀ
    any(isnan, Bᵀ) && println("NaN in tensor at time $T")

    return MPEM3(B)
end

prob_partial_msg_glauber(yₗᵗ, yₗ₁ᵗ, xₗᵗ) = ( yₗᵗ == yₗ₁ᵗ + potts2spin(xₗᵗ) )

function prob_ijy_glauber(xᵢᵗ⁺¹, xⱼᵗ, yᵗ, βJ, βh)
    h = βJ * (potts2spin(xⱼᵗ) + yᵗ) + βh
    p = exp(potts2spin(xᵢᵗ⁺¹) * h) / (2*cosh(h))
    @assert 0 ≤ p ≤ 1
    p
end

function onebpiter!(bp::MPBP{q,T,F,<:HomogeneousGlauberFactor}, i::Integer; 
    svd_trunc::SVDTrunc=TruncThresh(1e-6)) where {q,T,F}

    _onebpiter!(bp, i, f_bp_glauber; svd_trunc)
end

function magnetizations(bp::MPBP{q,T,F,<:GlauberFactor}) where {q,T,F}
    map(beliefs(bp)) do bᵢ
        reduce.(-, bᵢ)
    end
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