import InvertedIndices: Not

include("mpem.jl")

kron2() = AllOneTensor()

function kron2(A₁::Array{F,4}) where F
    @cast _[m₁, n₁, xᵢ, x₁] := A₁[m₁, n₁, x₁, xᵢ]
end

function kron2(A₁::Array{F,4}, A₂::Array{F,4}) where F
    @cast _[(m₁, m₂), (n₁, n₂), xᵢ, x₁, x₂] := A₁[m₁, n₁, x₁, xᵢ] * 
        A₂[m₂, n₂, x₂, xᵢ]
end
function kron2(A₁::Array{F,4}, A₂::Array{F,4}, A₃::Array{F,4}) where F
    @cast _[(m₁, m₂, m₃), (n₁, n₂, n₃), xᵢ, x₁, x₂, x₃] := 
        A₁[m₁, n₁, x₁, xᵢ] * A₂[m₂, n₂, x₂, xᵢ] * A₃[m₃, n₃, x₃, xᵢ]
end

function f_bp(A::Vector{MPEM2{q,T,F}}, pᵢ⁰, wᵢ, ϕᵢ, j_index::UnitRange) where {q,T,F}
    @assert length(pᵢ⁰) == q
    @assert length(wᵢ) == T
    @assert length(ϕᵢ) == T
    @assert j_index.start in eachindex(A)
    z = length(A)      # z = |∂i|
    x_neigs = Iterators.product(fill(1:q, z)...) .|> collect

    B = Vector{Array{F,5}}(undef, T+1)
    A⁰ = kron2([A[k][begin] for k in eachindex(A)[Not(j_index)]]...)
    nrows = size(A⁰, 1); ncols = size(A⁰, 2)
    B⁰ = zeros(q, q, nrows, ncols, q)
    
    for xᵢ¹ in 1:q, xᵢ⁰ in 1:q
        for xₙᵢ⁰ in x_neigs
            xⱼ⁰ = xₙᵢ⁰[j_index.start]
            xₙᵢ₋ⱼ⁰ = xₙᵢ⁰[Not(j_index)]
            # @show length(xₙᵢ₋ⱼ⁰)
            for a¹ in axes(A⁰, 2)
                B⁰[xᵢ⁰,xⱼ⁰,1,a¹,xᵢ¹] += wᵢ[1](xᵢ¹, xₙᵢ⁰,xᵢ⁰) *
                    A⁰[1,a¹,xᵢ⁰,xₙᵢ₋ⱼ⁰...]
            end
        end
        B⁰[xᵢ⁰,:,:,:,xᵢ¹] .*= ϕᵢ[1][xᵢ¹] * pᵢ⁰[xᵢ⁰] 
    end
    B[1] = B⁰

    for t in 1:T-1
        # select incoming A's but not the j-th one
        Aᵗ = kron2([A[k][begin+t] for k in eachindex(A)[Not(j_index)]]...)
        nrows = size(Aᵗ, 1); ncols = size(Aᵗ, 2)
        Bᵗ = zeros(q, q, nrows, ncols, q)

        for xᵢᵗ⁺¹ in 1:q
            for xᵢᵗ in 1:q
                for xₙᵢᵗ in x_neigs
                    xⱼᵗ = xₙᵢᵗ[j_index.start]
                    xₙᵢ₋ⱼᵗ = xₙᵢᵗ[Not(j_index)]
                    for aᵗ in axes(Aᵗ, 1), aᵗ⁺¹ in axes(Aᵗ, 2)
                        Bᵗ[xᵢᵗ,xⱼᵗ,aᵗ,aᵗ⁺¹,xᵢᵗ⁺¹] += wᵢ[t+1](xᵢᵗ⁺¹,xₙᵢᵗ,xᵢᵗ) *
                            Aᵗ[aᵗ,aᵗ⁺¹,xᵢᵗ,xₙᵢ₋ⱼᵗ...]
                    end
                end
            end
            Bᵗ[:,:,:,:,xᵢᵗ⁺¹] *= ϕᵢ[t+1][xᵢᵗ⁺¹]
        end
        B[t+1] = Bᵗ
    end

    Aᵀ = kron2([A[k][end] for k in eachindex(A)[Not(j_index)]]...)
    nrows = size(Aᵀ, 1); ncols = size(Aᵀ, 2)
    Bᵀ = ones(q, q, nrows, ncols, q)

    # if z==1 means j is the only neighbor of i and Bᵀ is uniform
    if z != 1 
        Aᵀ_reshaped = reshape(Aᵀ, size(Aᵀ)[1:3]..., prod(size(Aᵀ)[4:end]))
        Aᵀ_reshaped_summed = sum(Aᵀ_reshaped, dims=4)[:,1,:,1]
        # Bᵀ has some redundant dimensions, like the 5-th. We decided that it should 
        #  take the same values no matter the 5-th index
        # Same goes for the 2nd index, i.e. xⱼᵀ, on which Bᵀ does not depend.
        # Transpose to match indices aᵀ and xᵢᵀ.
        for xⱼᵀ in 1:q, xᵢᵀ⁺¹ in 1:q
            Bᵀ[:,xⱼᵀ,:,1,xᵢᵀ⁺¹] .= Aᵀ_reshaped_summed'
        end
    end
    B[end] = Bᵀ
    return MPEM3(B)
end