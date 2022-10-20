import Lazy: @forward
import TensorCast: @cast, @reduce, @matmul, TensorCast
import LinearAlgebra: svd, Diagonal, norm, tr
import Base.-
import ProgressMeter: Progress, next!
using Tullio: @tullio

abstract type MPEM; end

include("mpem2.jl")
include("mpem3.jl")

function normalize!(A::MPEM, method = norm)
    N = norm(A)
    for Aᵗ in A
        Aᵗ ./= N
    end
    A
end

function normalize_eachmatrix!(A::MPEM)
    for m in A
        mm = maximum(abs, m)
        if !any(isnan, mm) && !any(isinf, mm)
            m ./= mm
        end
    end
    A
end

-(A::T, B::T) where {T<:MPEM2} = MPEM2([AA .- BB for (AA,BB) in zip(A.tensors,B.tensors)])
-(A::T, B::T) where {T<:MPEM3} = MPEM3([AA .- BB for (AA,BB) in zip(A.tensors,B.tensors)])