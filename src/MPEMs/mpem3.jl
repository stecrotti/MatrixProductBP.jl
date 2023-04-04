# Matrix [Bᵗᵢⱼ(xᵢᵗ⁺¹,xᵢᵗ,xⱼᵗ)]ₘₙ is stored as a 5-array B[m,n,xᵢᵗ,xⱼᵗ,xᵢᵗ⁺¹]
# The last matrix should have the same values no matter what xᵢᵀ⁺¹ is
struct MPEM3{F<:Real} <: MPEM
    tensors::Vector{Array{F,5}}
    function MPEM3(tensors::Vector{Array{F,5}}) where {F<:Real}
        check_bond_dims(tensors) ||
            throw(ArgumentError("Matrix indices for matrix product non compatible"))
        new{F}(tensors)
    end
end

@forward MPEM3.tensors getindex, iterate, firstindex, lastindex, setindex!, 
    check_bond_dims, length, eachindex
    
getT(B::MPEM3) = length(B.tensors) - 1

# evaluate of MPEM3 is not simply evaluate(MatrixProductTrain{F,5}) because of how we
# interpret the entries of the tensors
function evaluate(B::MPEM3, x)
    length(x) == length(B) || throw(ArgumentError("`x` must be of length $(length(B)), got $(length(x))"))
    M = B[begin][:, :, x[1][1], x[1][2], x[2][1]]
    for t in 2:lastindex(B)-1
        M = M * B[t][:, :, x[t][1], x[t][2], x[t+1][1]]
    end
    M = M * B[end][:, :, x[end][1], x[end][2], x[1][1]]
    return tr(M)
end

# convert mpem3 into mpem2 via a Left to Right sweep of SVD's
function mpem2(B::MPEM3{F}) where {F}

    C = Vector{Array{F,4}}(undef, length(B))
    qᵢᵗ = size(B[1], 3); qⱼᵗ = size(B[1], 4); qᵢᵗ⁺¹ = size(B[1], 5)

    B⁰ = B[begin]
    @cast M[(xᵢᵗ, xⱼᵗ, m), (n, xᵢᵗ⁺¹)] |= B⁰[m, n, xᵢᵗ, xⱼᵗ, xᵢᵗ⁺¹]
    U, λ, V = svd(M) 
    Bᵗ⁺¹_new = fill(1.0,1,1,1,1)  # initialize
    for t in 1:getT(B)  
        M = length(λ)     
        @cast Cᵗ[m, k, xᵢᵗ, xⱼᵗ] := U[(xᵢᵗ, xⱼᵗ, m), k] k in 1:M, xᵢᵗ in 1:qᵢᵗ, xⱼᵗ in 1:qⱼᵗ
        C[t] = Cᵗ
        @cast Vt[m, n, xᵢᵗ⁺¹] := V'[m, (n, xᵢᵗ⁺¹)]  xᵢᵗ⁺¹ in 1:qᵢᵗ⁺¹
        Bᵗ⁺¹ = B[t+1]
        @tullio Bᵗ⁺¹_new[m, n, xᵢᵗ⁺¹, xⱼᵗ⁺¹, xᵢᵗ⁺²] := λ[m] * 
            Vt[m, l, xᵢᵗ⁺¹] * Bᵗ⁺¹[l, n, xᵢᵗ⁺¹, xⱼᵗ⁺¹, xᵢᵗ⁺²] 
        @cast M[(xᵢᵗ⁺¹, xⱼᵗ⁺¹, m), (n, xᵢᵗ⁺²)] |= Bᵗ⁺¹_new[m, n, xᵢᵗ⁺¹, xⱼᵗ⁺¹,xᵢᵗ⁺²]
        U, λ, V = svd(M)
    end
    M = length(λ)     
    @cast Cᵀ[m, k, xᵢᵀ, xⱼᵀ] := U[(xᵢᵀ, xⱼᵀ, m), k] k in 1:M, xᵢᵀ in 1:qᵢᵗ, xⱼᵀ in 1:qⱼᵗ
    C[end] = Cᵀ
    @cast Vt[m, n, xᵢ⁰] := V'[m, (n, xᵢ⁰)]  xᵢ⁰ in 1:qᵢᵗ
    C⁰ = C[begin]
    @tullio C⁰_new[m, n, xᵢ⁰, xⱼ⁰] := λ[m] * 
            Vt[m, l, xᵢ⁰] * C⁰[l, n, xᵢ⁰, xⱼ⁰]
    C[begin] = C⁰_new
    return MPEM2(C)
end