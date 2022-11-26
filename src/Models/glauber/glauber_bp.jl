const q_glauber = 2

struct GenericGlauberFactor{T<:Real}  <: BPFactor 
    βJ :: Vector{T}      
    βh :: T
end

struct HomogeneousGlauberFactor{T<:Real} <: SimpleBPFactor 
    βJ :: T     
    βh :: T
end

getq(::Type{<:GenericGlauberFactor}) = q_glauber
getq(::Type{<:HomogeneousGlauberFactor}) = q_glauber

function HomogeneousGlauberFactor(J::T, h::T, β::T) where {T<:Real}
    HomogeneousGlauberFactor(J*β, h*β)
end

function GenericGlauberFactor(J::Vector{T}, h::T, β::T) where {T<:Real}
    GenericGlauberFactor(J.*β, h*β)
end

function (fᵢ::GenericGlauberFactor)(xᵢᵗ⁺¹::Integer, 
        xₙᵢᵗ::AbstractVector{<:Integer}, 
        xᵢᵗ::Integer)
    @assert xᵢᵗ⁺¹ ∈ 1:q_glauber
    @assert all(x ∈ 1:q_glauber for x in xₙᵢᵗ)
    @assert length(xₙᵢᵗ) == length(fᵢ.βJ)

    hⱼᵢ = sum( Jᵢⱼ * potts2spin(xⱼᵗ) for (xⱼᵗ,Jᵢⱼ) in zip(xₙᵢᵗ, fᵢ.βJ))
    E = - potts2spin(xᵢᵗ⁺¹) * (hⱼᵢ + fᵢ.βh)
    exp( -E ) / (2cosh(E))
end

function (fᵢ::HomogeneousGlauberFactor)(xᵢᵗ⁺¹::Integer, 
        xₙᵢᵗ::AbstractVector{<:Integer}, 
        xᵢᵗ::Integer)
    @assert xᵢᵗ⁺¹ ∈ 1:q_glauber
    @assert all(x ∈ 1:q_glauber for x in xₙᵢᵗ)

    hⱼᵢ = sum( Jᵢⱼ * potts2spin(xⱼᵗ) for (xⱼᵗ,Jᵢⱼ) in zip(xₙᵢᵗ, fᵢ.βJ))
    hⱼᵢ = fᵢ.βJ * sum(potts2spin, xₙᵢᵗ)
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
# seems to be type stable
function glauber_factors(ising::Ising, T::Integer)
    map(1:nv(ising.g)) do i
        ei = outedges(ising.g, i)
        ∂i = idx.(ei)
        J = ising.J[∂i]
        h = ising.h[i]
        wᵢᵗ = if is_homogeneous(ising)
            HomogeneousGlauberFactor(J[1], h, ising.β)
        else
            GenericGlauberFactor(J, h, ising.β)
        end
        fill(wᵢᵗ, T)
    end
end

idx_to_value(x::Integer, ::Type{<:GenericGlauberFactor}) = potts2spin(x)
idx_to_value(x::Integer, ::Type{<:HomogeneousGlauberFactor}) = potts2spin(x)

prob_partial_msg_glauber(yₗᵗ, yₗ₁ᵗ, xₗᵗ) = ( yₗᵗ == ( yₗ₁ᵗ + potts2spin(xₗᵗ) ) )

# the sum of n spins can be one of (n+1) values. We sort them increasingly and
#  index them by k
function _idx_map(n::Integer, k::Integer) 
    @assert n ≥ 0
    @assert k ∈ 1:(n+1)
    return - n + 2*(k-1)
end

# compute message m(i→j, l) from m(i→j, l-1) 
# returns an `MPEM2` [Aᵗᵢⱼ,ₗ(yₗᵗ,xᵢᵗ)]ₘₙ is stored as a 4-array A[m,n,yₗᵗ,xᵢᵗ]
function f_bp_partial(mₗᵢ::MPEM2{q,T,F}, mᵢⱼₗ₁::MPEM2{q1,T,F}, 
        wᵢ::Vector{U}, l::Integer) where {q,q1,T,F,U<:HomogeneousGlauberFactor}
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
function f_bp_partial_ij(A::MPEM2{Q,T,F}, pᵢ⁰, wᵢ::Vector{U}, ϕᵢ, 
        d::Integer; prob = prob_ijy(U)) where {Q,T,F,U<:HomogeneousGlauberFactor}
    q = getq(U)
    B = Vector{Array{F,5}}(undef, T+1)

    A⁰ = A[begin]
    nrows = size(A⁰, 1); ncols = size(A⁰, 2)
    B⁰ = zeros(q, q, nrows, ncols, q)

    for xᵢ⁰ in 1:q
        for xᵢ¹ in 1:q
            for z⁰ in 1:(d+1)
                y⁰ = _idx_map(d, z⁰)
                for xⱼ⁰ in 1:q
                    p = prob(xᵢ¹, xⱼ⁰, y⁰, wᵢ[begin].βJ, wᵢ[begin].βh)
                    B⁰[xᵢ⁰,xⱼ⁰,1,:,xᵢ¹] .+= p * A⁰[1,:,z⁰,xᵢ⁰]
                end
            end
        end
        B⁰[xᵢ⁰,:,:,:,:] .*= pᵢ⁰[xᵢ⁰]  * ϕᵢ[begin][xᵢ⁰]
    end
    B[begin] = B⁰

    for t in 1:T-1
        Aᵗ = A[begin+t]
        nrows = size(Aᵗ, 1); ncols = size(Aᵗ, 2)
        Bᵗ = zeros(q, q, nrows, ncols, q)

        for xᵢᵗ in 1:q
            for xᵢᵗ⁺¹ in 1:q
                for xⱼᵗ in 1:q
                    for zᵗ in 1:(d+1)
                        yᵗ = _idx_map(d, zᵗ)
                        p = prob(xᵢᵗ⁺¹, xⱼᵗ, yᵗ, wᵢ[t+1].βJ, wᵢ[t+1].βh)
                        Bᵗ[xᵢᵗ,xⱼᵗ,:,:,xᵢᵗ⁺¹] .+= p * Aᵗ[:,:,zᵗ,xᵢᵗ]
                    end
                end
            end
            Bᵗ[xᵢᵗ,:,:,:,:] *= ϕᵢ[t+1][xᵢᵗ]
        end
        any(isnan, Bᵗ) && println("NaN in tensor at time $t")
        B[begin+t] = Bᵗ
    end

    Aᵀ = A[end]
    nrows = size(Aᵀ, 1); ncols = size(Aᵀ, 2)
    Bᵀ = zeros(q, q, nrows, ncols, q)

    for xᵢᵀ in 1:q
        for xᵢᵀ⁺¹ in 1:q
            for xⱼᵀ in 1:q
                for zᵀ in 1:(d+1)
                    Bᵀ[xᵢᵀ,xⱼᵀ,:,:,xᵢᵀ⁺¹] .+= Aᵀ[:,:,zᵀ,xᵢᵀ]
                end
            end
        end
        Bᵀ[xᵢᵀ,:,:,:,:] *= ϕᵢ[end][xᵢᵀ]
    end
    B[end] = Bᵀ
    any(isnan, Bᵀ) && println("NaN in tensor at time $T")

    return MPEM3(B)
end

function prob_ijy(::Type{<:HomogeneousGlauberFactor})
    function prob_ijy_glauber(xᵢᵗ⁺¹, xⱼᵗ, yᵗ, βJ, βh)
        h = βJ * (potts2spin(xⱼᵗ) + yᵗ) + βh
        p = exp(potts2spin(xᵢᵗ⁺¹) * h) / (2*cosh(h))
        @assert 0 ≤ p ≤ 1
        p
    end
end

function prob_ijy_dummy(::Type{<:HomogeneousGlauberFactor})
    # ignore neighbor because it doesn't exist
    function prob_ijy_dummy_glauber(xᵢᵗ⁺¹, xⱼᵗ, yᵗ, βJ, βh)
        h = βJ * yᵗ + βh
        p = exp(potts2spin(xᵢᵗ⁺¹) * h) / (2*cosh(h))
        @assert 0 ≤ p ≤ 1
        p
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

function glauber_infinite_graph(T::Integer, k::Integer, pᵢ⁰;
        β::Real=1.0, J::Real=1.0, h::Real=0.0,
        svd_trunc::SVDTrunc=TruncThresh(1e-6), maxiter=5, tol=1e-5,
        showprogress=true)
    wᵢ = fill(HomogeneousGlauberFactor(J, h, β), T)
    A, maxiter, Δs = iterate_bp_infinite_graph(T, k, pᵢ⁰, wᵢ; 
        svd_trunc, maxiter, tol, showprogress)
end