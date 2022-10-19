# Matrix [Bᵗᵢⱼ(xᵢᵗ⁺¹,xᵢᵗ,xⱼᵗ)]ₘₙ is stored as a 5-array B[xᵢᵗ,xⱼᵗ,m,n,xᵢᵗ⁺¹]
# T is the final time
# The last matrix should have the same values no matter what xᵢᵀ⁺¹ is
struct MPEM3{q,T,F<:Real} <: MPEM
    tensors :: Vector{Array{F,5}}     # Vector of length T+1
    function MPEM3(tensors::Vector{Array{F,5}}) where {F<:Real}
        T = length(tensors)-1
        q = size(tensors[1],1)
        @assert all(size(t,1) == size(t,2) == size(t,5) == q for t in tensors)
        @assert size(tensors[1],3) == size(tensors[end],4) == 1
        @assert check_bond_dims3(tensors)
        t = tensors[end][:,:,:,:,1]
        @assert all(tensors[end][:,:,:,:,x] == t for x in 2:q) "$([tensors[end][:,:,:,:,x] for x in 1:q])"
        new{q,T,F}(tensors)
    end
end

function check_bond_dims3(tensors::Vector{<:Array})
    for t in 1:lastindex(tensors)-1
        dᵗ = size(tensors[t],4)
        dᵗ⁺¹ = size(tensors[t+1],3)
        if dᵗ != dᵗ⁺¹
            println("Bond size for matrix t=$t. dᵗ=$dᵗ, dᵗ⁺¹=$dᵗ⁺¹")
            return false
        end
    end
    return true
end


function bond_dims(B::MPEM3{q,T,F}) where {q,T,F}
    return [size(B[t], 4) for t in 1:lastindex(B)-1]
end

@forward MPEM3.tensors Base.getindex, Base.iterate, Base.firstindex, 
Base.lastindex, Base.setindex!,
check_bond_dims3

getq(::MPEM3{q,T,F}) where {q,T,F} = q
getT(::MPEM3{q,T,F}) where {q,T,F} = T
Base.eltype(::MPEM3{q,T,F}) where {q,T,F} = F

function evaluate(B::MPEM3{q,T,F}, x) where {q,T,F}
    @assert length(x) == T + 1
    @assert all(xx[1] ∈ 1:q && xx[2] ∈ 1:q for xx in x)
    M = [1.0;;]
    for t in 1:lastindex(B)-1
        M = M * B[t][x[t][1], x[t][2], :, :, x[t+1][1]]
    end
    M = M * B[end][x[end][1], x[end][2], :, :, 1]
    return only(M)
end

# convert mpem2 into mpem3 via a Left to Right sweep of SVD's
function mpem2(B::MPEM3{q,T,F}; showprogress=false) where {q,T,F}
    C = Vector{Array{F,4}}(undef, T+1)

    B⁰ = B[begin]
    @cast M[(xᵢᵗ, xⱼᵗ, m), (n, xᵢᵗ⁺¹)] |= B⁰[xᵢᵗ, xⱼᵗ, m, n, xᵢᵗ⁺¹]
    Bᵗ⁺¹_new = rand(1,1,1,1)  # initialize

    dt = showprogress ? 1.0 : Inf
    prog = Progress(T, dt=dt, desc="Converting B to C")
    for t in 1:T
        U, λ, V = svd(M)   
        m = length(λ)     
        @cast Cᵗ[m, k, xᵢ, xⱼ] := U[(xᵢ, xⱼ, m), k] k in 1:m, xᵢ in 1:q, xⱼ in 1:q
        C[t] = Cᵗ
        @cast Vt[m, n, xᵢᵗ⁺¹] := V'[m, (n, xᵢᵗ⁺¹)]  xᵢᵗ⁺¹ in 1:q

        Bᵗ⁺¹ = B[t+1]
        @reduce Bᵗ⁺¹_new[xᵢᵗ⁺¹, xⱼᵗ⁺¹, m, n, xᵢᵗ⁺²] |= sum(l) λ[m] * 
            Vt[m, l, xᵢᵗ⁺¹] * Bᵗ⁺¹[xᵢᵗ⁺¹, xⱼᵗ⁺¹, l, n, xᵢᵗ⁺²] 
        @cast M[(xᵢᵗ⁺¹, xⱼᵗ⁺¹, m), (n, xᵢᵗ⁺²)] |= Bᵗ⁺¹_new[xᵢᵗ⁺¹, xⱼᵗ⁺¹, m, n, xᵢᵗ⁺²]
        next!(prog, showvalues=[(:t,"$t/$T")])
    end
    @cast Cᵀ[m,n,xᵢ,xⱼ] := Bᵗ⁺¹_new[xᵢ,xⱼ,m,n,1]
    C[end] = Cᵀ
    @assert check_bond_dims2(C)
    return MPEM2(C)
end