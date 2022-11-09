include("bp.jl")
include("mpdbp.jl")

#### SIS

# compute outgoing message efficiently for any degree
# return a `MPMEM3` just like `f_bp`
function f_bp_sis(A::Vector{MPEM2{q,T,F}}, pᵢ⁰, wᵢ, ϕᵢ, ψₙᵢ, j::Integer;
        svd_trunc=TruncThresh(1e-6)) where {q,T,F}
    d = length(A) - 1
    λ = wᵢ[1].λ; @assert all(wᵢᵗ.λ == λ for wᵢᵗ in wᵢ)
    κ = wᵢ[1].κ; @assert all(wᵢᵗ.κ == κ for wᵢᵗ in wᵢ)

    # initialize recursion
    M = reshape([1.0 1; 0 0], (1,1,q,q))
    mᵢⱼₗ₁ = MPEM2( fill(M, T+1) )

    for k in eachindex(A)
        k == j && continue
        mᵢⱼₗ₁ = f_bp_partial_sis(A[k], mᵢⱼₗ₁, λ)
        # SVD L to R with no truncation
        sweep_LtoR!(mᵢⱼₗ₁, svd_trunc=TruncThresh(0.0))
        # SVD R to L with truncations
        sweep_RtoL!(mᵢⱼₗ₁; svd_trunc)
    end

    # combine the last partial message with p(xᵢᵗ⁺¹|xᵢᵗ, xⱼᵗ, yᵗ)
    B = f_bp_partial_ij_sis(mᵢⱼₗ₁, λ, κ, pᵢ⁰, ϕᵢ)
    return B
end

# compute message m(i→j, l) from m(i→j, l-1) 
# returns an `MPEM2` [Aᵗᵢⱼ,ₗ(yₗᵗ,xᵢᵗ)]ₘₙ is stored as a 4-array A[m,n,yₗᵗ,xᵢᵗ]
function f_bp_partial_sis(mₗᵢ::MPEM2{q,T,F}, mᵢⱼₗ₁::MPEM2{q,T,F}, 
        λ::Real) where {q,T,F}
    @assert q==q_sis
    AA = Vector{Array{F,4}}(undef, T+1)

    for t in eachindex(AA)
        Aᵗ = kron2(mₗᵢ[t], mᵢⱼₗ₁[t])
        nrows = size(Aᵗ, 1); ncols = size(Aᵗ, 2)
        AAᵗ = zeros(nrows, ncols, q, q)
        for yₗᵗ in 1:q
            for xᵢᵗ in 1:q
                for yₗ₁ᵗ in 1:q
                    for xₗᵗ in 1:q
                        p = prob_partial_msg_sis(yₗᵗ, yₗ₁ᵗ, xₗᵗ, λ)
                        AAᵗ[:,:,yₗᵗ,xᵢᵗ] .+= p * Aᵗ[:,:,xᵢᵗ,xₗᵗ,yₗ₁ᵗ] 
                    end
                end
            end
        end
        AA[t] = AAᵗ
    end

    return MPEM2(AA)
end

# compute m(i→j) from m(i→j,d)
function f_bp_partial_ij_sis(A::MPEM2{q,T,F}, λ::Real, κ::Real, 
        pᵢ⁰, ϕᵢ) where {q,T,F}

    B = Vector{Array{F,5}}(undef, T+1)

    A⁰ = A[begin]
    nrows = size(A⁰, 1); ncols = size(A⁰, 2)
    B⁰ = zeros(q, q, nrows, ncols, q)
    
    for xᵢ¹ in 1:q, xᵢ⁰ in 1:q
        for y⁰ in 1:q
            for xⱼ⁰ in 1:q
                p = prob_ijy_sis(xᵢ¹, xᵢ⁰,xⱼ⁰, y⁰, λ, κ)
                B⁰[xᵢ⁰,xⱼ⁰,1,:,xᵢ¹] .+= p * A⁰[1,:,y⁰,xᵢ⁰]
            end
        end
        B⁰[xᵢ⁰,:,:,:,xᵢ¹] .*= ϕᵢ[1][xᵢ¹] * pᵢ⁰[xᵢ⁰] 
    end
    B[begin] = B⁰

    for t in 1:T-1
        Aᵗ = A[begin+t]
        nrows = size(Aᵗ, 1); ncols = size(Aᵗ, 2)
        Bᵗ = zeros(q, q, nrows, ncols, q)

        for xᵢᵗ⁺¹ in 1:q
            for xᵢᵗ in 1:q
                for xⱼᵗ in 1:q
                    for yᵗ in 1:q
                        p = prob_ijy_sis(xᵢᵗ⁺¹, xᵢᵗ, xⱼᵗ, yᵗ, λ, κ)
                        Bᵗ[xᵢᵗ,xⱼᵗ,:,:,xᵢᵗ⁺¹] .+= p * Aᵗ[:,:,yᵗ,xᵢᵗ]
                    end
                end
            end
            Bᵗ[:,:,:,:,xᵢᵗ⁺¹] *= ϕᵢ[t+1][xᵢᵗ⁺¹]
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
                for yᵀ in 1:q
                    Bᵀ[xᵢᵀ,xⱼᵀ,:,:,xᵢᵀ⁺¹] .+= Aᵀ[:,:,yᵀ,xᵢᵀ]
                end
            end
        end
    end
    B[end] = Bᵀ
    any(isnan, Bᵀ) && println("NaN in tensor at time $T")

    return MPEM3(B)
end

function prob_partial_msg_sis(yₖ, yₖ₁, xₖ, λ)
    if yₖ == I
        return 1 - (yₖ₁==S)*(1-λ*(xₖ==I))
    elseif yₖ == S
        return (yₖ₁==S)*(1-λ*(xₖ==I))
    end
end

function prob_ijy_sis(xᵢᵗ⁺¹, xᵢᵗ, xⱼᵗ, yᵗ, λ, κ)
    z = 1 - λ*(xⱼᵗ==I)
    w = (yᵗ==S)
    if xᵢᵗ⁺¹ == I
        return (xᵢᵗ==I) * (1 - κ) + (xᵢᵗ==S) * (1 - z * w) 
    elseif xᵢᵗ⁺¹ == S
        return (xᵢᵗ==I) * κ + (xᵢᵗ==S) * z * w
    end
    error("shouldn't be here")
    return -Inf
end

function onebpiter!(bp::MPdBP{q,T,F,<:SISFactor}, i::Integer; 
        svd_trunc::SVDTrunc=TruncThresh(1e-6)) where {q,T,F}
    @unpack g, w, ϕ, ψ, p⁰, μ = bp
    ein = inedges(g,i)
    eout = outedges(g, i)
    A = μ[ein.|>idx]
    @assert all(normalization(a) ≈ 1 for a in A)
    zᵢ = 1.0
    for (j_ind, e_out) in enumerate(eout)
        B = f_bp_sis(A, p⁰[i], w[i], ϕ[i], ψ[eout.|>idx], j_ind; svd_trunc)
        C = mpem2(B)
        μ[idx(e_out)] = sweep_RtoL!(C; svd_trunc)
        zᵢ₂ⱼ = normalize!(μ[idx(e_out)])
        zᵢ *= zᵢ₂ⱼ
    end
    dᵢ = length(ein)
    return zᵢ ^ (1 / dᵢ)
end

#### Glauber

# the sum of n spins can be one of (n+1) values. We sort them increasingly and
#  index them by k
function idx_map(n::Integer, k::Integer) 
    @assert n ≥ 0
    @assert k ∈ 1:(n+1)
    return - n + 2*(k-1)
end

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

    # if l==1 && d>0
    #     x = evaluate(mᵢⱼₗ₁, [[1,1] for _ in 0:T])
    #     y = evaluate(A[findfirst(!isequal(j), eachindex(A))], [[1,1] for _ in 0:T])
    #     @assert x ≈ y
    # end
    
    # combine the last partial message with p(xᵢᵗ⁺¹|xᵢᵗ, xⱼᵗ, yᵗ)
    B = f_bp_partial_ij_glauber(mᵢⱼₗ₁, βJ, βh, pᵢ⁰, ϕᵢ, d)

    return B
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
            yₗᵗ = idx_map(l, zₗᵗ)
            for xᵢᵗ in 1:q
                for zₗ₁ᵗ in 1:l
                    yₗ₁ᵗ = idx_map(l-1, zₗ₁ᵗ) 
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
            y⁰ = idx_map(d, z⁰)
            for xⱼ⁰ in 1:q
                p = prob_ijy_glauber(xᵢ¹, xⱼ⁰, y⁰, βJ, βh)
                B⁰[xᵢ⁰,xⱼ⁰,1,:,xᵢ¹] .+= p * A⁰[1,:,z⁰,xᵢ⁰]
            end
        end
        B⁰[xᵢ⁰,:,:,:,xᵢ¹] .*= ϕᵢ[1][xᵢ¹] * pᵢ⁰[xᵢ⁰] 
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
                        yᵗ = idx_map(d, zᵗ)
                        p = prob_ijy_glauber(xᵢᵗ⁺¹, xⱼᵗ, yᵗ, βJ, βh)
                        Bᵗ[xᵢᵗ,xⱼᵗ,:,:,xᵢᵗ⁺¹] .+= p * Aᵗ[:,:,zᵗ,xᵢᵗ]
                    end
                end
            end
            Bᵗ[:,:,:,:,xᵢᵗ⁺¹] *= ϕᵢ[t+1][xᵢᵗ⁺¹]
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
                    yᵀ = idx_map(d, zᵀ)
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

function onebpiter!(bp::MPdBP{q,T,F,<:GlauberFactor}, i::Integer; 
        svd_trunc::SVDTrunc=TruncThresh(1e-6)) where {q,T,F}
    @unpack g, w, ϕ, ψ, p⁰, μ = bp
    ein = inedges(g,i)
    eout = outedges(g, i)
    A = μ[ein.|>idx]
    @assert all(normalization(a) ≈ 1 for a in A)
    zᵢ = 1.0
    for (j_ind, e_out) in enumerate(eout)
        B = f_bp_glauber(A, p⁰[i], w[i], ϕ[i], ψ[eout.|>idx], j_ind; svd_trunc)
        C = mpem2(B)
        μ[idx(e_out)] = sweep_RtoL!(C; svd_trunc)
        zᵢ₂ⱼ = normalize!(μ[idx(e_out)])
        zᵢ *= zᵢ₂ⱼ
    end
    dᵢ = length(ein)
    return zᵢ ^ (1 / dᵢ)
end