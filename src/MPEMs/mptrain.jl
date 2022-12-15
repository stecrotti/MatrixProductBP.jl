struct MatrixProductTrain{F<:Real, N} <: MPEM
    tensors::Vector{Array{F,N}}
    function MatrixProductTrain(tensors::Vector{Array{F,N}}) where {F<:Real, N}
        size(tensors[1],1) == size(tensors[end],2) == 1 ||
            throw(ArgumentError("First matrix must have 1 row, last matrix must have 1 column"))
        check_bond_dims(tensors) ||
            throw(ArgumentError("Matrix indices for matrix product non compatible"))
        new{F,N}(tensors)
    end
end

@forward MatrixProductTrain.tensors getindex, iterate, firstindex, lastindex, setindex!, 
    check_bond_dims, length


check_bond_dims(A::MatrixProductTrain) = check_bond_dims(A.tensors)

function check_bond_dims(tensors::Vector{<:Array})
    for t in 1:lastindex(tensors)-1
        dᵗ = size(tensors[t],2)
        dᵗ⁺¹ = size(tensors[t+1],1)
        if dᵗ != dᵗ⁺¹
            println("Bond size for matrix t=$t. dᵗ=$dᵗ, dᵗ⁺¹=$dᵗ⁺¹")
            return false
        end
    end
    return true
end

# keep size of matrix elements under control by dividing by the max
# return the log of the product of the individual normalizations 
function normalize_eachmatrix!(A::MatrixProductTrain)
    c = 0.0
    for m in A
        mm = maximum(abs, m)
        if !any(isnan, mm) && !any(isinf, mm)
            m ./= mm
            c += log(mm)
        end
    end
    c
end

-(A::T, B::T) where {T<:MatrixProductTrain} = MPEM([AA .- BB for (AA,BB) in zip(A.tensors,B.tensors)])

isapprox(A::T, B::T; kw...) where {T<:MatrixProductTrain} = isapprox(A.tensors, B.tensors; kw...)

const MPEM1{F} = MatrixProductTrain{F, 3}
const MPEM2{F} = MatrixProductTrain{F, 4}
const MPEM3{F} = MatrixProductTrain{F, 5}

"Construct a uniform mpem with given bond dimensions"
mpem(T::Int, d::Int, bondsizes, q...) = MatrixProductTrain([ones(bondsizes[t], bondsizes[t+1], q...) for t in 1:T+1])

"Construct a random mpem with given bond dimensions"
rand_mpem(T::Int; d::Int, bondsizes, q...) = MatrixProductTrain([rand(bondsizes[t], bondsizes[t+1], q...) for t in 1:T+1])

bond_dims(A::MPEM) = [size(A[t], 2) for t in 1:lastindex(A)-1]

getT(A::MatrixProductTrain) = length(A) - 1

eltype(::MatrixProductTrain{N,F}) where {N,F} = F

evaluate(A::MatrixProductTrain, X...) = only(prod(@view a[:, :, x...] for (a,x) in zip(A, X...)))

