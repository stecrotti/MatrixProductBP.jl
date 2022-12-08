# Matrix [Bᵗᵢⱼ(xᵢᵗ⁺¹,xᵢᵗ,xⱼᵗ)]ₘₙ is stored as a 5-array B[xᵢᵗ,xⱼᵗ,m,n,xᵢᵗ⁺¹]
# The last matrix should have the same values no matter what xᵢᵀ⁺¹ is
struct MPEM3{F<:Real} <: MPEM
    tensors :: Vector{Array{F,5}}     # Vector of length T+1
    function MPEM3(tensors::Vector{Array{F,5}}) where {F<:Real}
        check_bond_dims3(tensors) ||
            throw(ArgumentError("Matrix indices for matrix product non compatible"))
        
        # tensor Bᵀ does not actually depend on its last index. For simplicity, we ensure it takes the same value no matter the value of the last index
        t = tensors[end][:,:,:,:,1]
        @assert all(tensors[end][:,:,:,:,x] == t for x in axes(tensors[end],5))
        new{F}(tensors)
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

function bond_dims(B::MPEM3)
    return [size(B[t], 4) for t in 1:lastindex(B)-1]
end

@forward MPEM3.tensors getindex, iterate, firstindex, lastindex, setindex!, 
    check_bond_dims3, length

getT(B::MPEM3)= length(B) - 1
eltype(::MPEM3{F}) where {F} = F

function evaluate(B::MPEM3, x)
    length(x) == length(B) || throw(ArgumentError("`x` must be of length $(length(B)), got $(length(x))"))
    M = [1.0;;]
    for t in 1:lastindex(B)-1
        M = M * B[t][x[t][1], x[t][2], :, :, x[t+1][1]]
    end
    M = M * B[end][x[end][1], x[end][2], :, :, 1]
    return only(M)
end

# convert mpem2 into mpem3 via a Left to Right sweep of SVD's
function mpem2(B::MPEM3{F}) where {F}
    C = Vector{Array{F,4}}(undef, length(B))
    qᵢᵗ = size(B[1], 1); qⱼᵗ = size(B[1], 2); qᵢᵗ⁺¹ = size(B[1], 5)

    B⁰ = B[begin]
    @cast M[(xᵢᵗ, xⱼᵗ, m), (n, xᵢᵗ⁺¹)] |= B⁰[xᵢᵗ, xⱼᵗ, m, n, xᵢᵗ⁺¹]
    Bᵗ⁺¹_new = rand(1,1,1,1)  # initialize
    for t in 1:getT(B)
        U, λ, V = svd(M)   
        m = length(λ)     
        @cast Cᵗ[m, k, xᵢᵗ, xⱼᵗ] := U[(xᵢᵗ, xⱼᵗ, m), k] k in 1:m, xᵢᵗ in 1:qᵢᵗ, xⱼᵗ in 1:qⱼᵗ
        C[t] = Cᵗ
        @cast Vt[m, n, xᵢᵗ⁺¹] := V'[m, (n, xᵢᵗ⁺¹)]  xᵢᵗ⁺¹ in 1:qᵢᵗ⁺¹
        Bᵗ⁺¹ = B[t+1]
        @tullio Bᵗ⁺¹_new[xᵢᵗ⁺¹, xⱼᵗ⁺¹, m, n, xᵢᵗ⁺²] := λ[m] * 
            Vt[m, l, xᵢᵗ⁺¹] * Bᵗ⁺¹[xᵢᵗ⁺¹, xⱼᵗ⁺¹, l, n, xᵢᵗ⁺²] 
        @cast M[(xᵢᵗ⁺¹, xⱼᵗ⁺¹, m), (n, xᵢᵗ⁺²)] |= Bᵗ⁺¹_new[xᵢᵗ⁺¹, xⱼᵗ⁺¹, m, n, xᵢᵗ⁺²]
    end
    @cast Cᵀ[m,n,xᵢ,xⱼ] := Bᵗ⁺¹_new[xᵢ,xⱼ,m,n,1]
    C[end] = Cᵀ
    @assert check_bond_dims2(C)
    return MPEM2(C)
end