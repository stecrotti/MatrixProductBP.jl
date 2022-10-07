import InvertedIndices: Not

include("mpem.jl")

function kron2(A₁::Array{F,4}, A₂::Array{F,4}) where F
    @cast _[(m₁, m₂), (n₁, n₂), xᵢ, x₁, x₂] := A₁[m₁, n₁, x₁, xᵢ] * 
        A₂[m₂, n₂, x₂, xᵢ]
end
function kron2(A₁::Array{F,4}, A₂::Array{F,4}, A₃::Array{F,4}) where F
    @cast _[(m₁, m₂, m₃), (n₁, n₂, n₃), xᵢ, x₁, x₂, x₃] := 
        A₁[m₁, n₁, x₁, xᵢ] * A₂[m₂, n₂, x₂, xᵢ] * A₃[m₃, n₃, x₃, xᵢ]
end

function f_bp(A::Vector{MPEM2{q,T,F}}, pᵢ, wᵢ, ϕᵢ, j_index) where {q,T,F}
    @assert length(pᵢ) == q
    @assert length(wᵢ) == T
    @assert length(ϕᵢ) == T
    z = length(A) + 1      # z = |∂i|
    x_neigs = Iterators.product(fill(1:q, z-1)...) .|> collect

    B = Vector{Array{F,5}}(undef, T+1)
    A⁰ = kron2([Aₖ[begin] for Aₖ in A]...)
    nrows = size(A⁰, 1); ncols = size(A⁰, 2)
    B⁰ = zeros(q, q, nrows, ncols, q)
    
    for xᵢ¹ in 1:q, xᵢ⁰ in 1:q
        for xₙᵢ⁰ in x_neigs
            xⱼ⁰ = xₙᵢ⁰[j_index]
            xₙᵢ₋ⱼ⁰ = xₙᵢ⁰[Not(j_index)]
            for a¹ in axes(A⁰, 2)
                B⁰[xᵢ⁰,xⱼ⁰,1,a¹,xᵢ¹] += wᵢ[1](xᵢ¹, xₙᵢ⁰,xᵢ⁰) *
                    A⁰[1,a¹,xᵢ⁰,xₙᵢ₋ⱼ⁰...]
            end
        end
        B⁰[xᵢ⁰,:,:,:,xᵢ¹] .*= ϕᵢ[1][xᵢ¹] * pᵢ[xᵢ⁰] 
    end
    B[1] = B⁰

    for t in 1:T-1
        Aᵗ = kron2([Aₖ[begin+t] for Aₖ in A]...)
        nrows = size(Aᵗ, 1); ncols = size(Aᵗ, 2)
        Bᵗ = zeros(q, q, nrows, ncols, q)

        for xᵢᵗ⁺¹ in 1:q
            for xᵢᵗ in 1:q
                for xₙᵢᵗ in x_neigs
                    xⱼᵗ = xₙᵢᵗ[j_index]
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

    Aᵀ = kron2([Aₖ[end] for Aₖ in A]...)
    nrows = size(Aᵀ, 1); ncols = size(Aᵀ, 2)
    Bᵀ = zeros(q, q, nrows, ncols, q)
    Aᵀ_reshaped = reshape(Aᵀ, size(Aᵀ)[1:3]..., prod(size(Aᵀ)[4:end]))
    Aᵀ_reshaped_summed = sum(Aᵀ_reshaped, dims=4)[:,1,:,1]
    for xⱼᵀ in 1:q, xᵢᵀ⁺¹ in 1:q
        Bᵀ[:,xⱼᵀ,:,1,xᵢᵀ⁺¹] .= Aᵀ_reshaped_summed'
    end
    B[end] = Bᵀ
    return B
end