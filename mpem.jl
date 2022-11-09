import Lazy: @forward
import TensorCast: @cast, @reduce, @matmul, TensorCast
import LinearAlgebra: svd, Diagonal, norm, tr
import Base.-
import ProgressMeter: Progress, next!
using Tullio: @tullio

abstract type MPEM; end

include("mpem2.jl")
include("mpem3.jl")

# keep size of matrix elements under control by dividing by the max
# return the log of the product of the individual normalizations 
function normalize_eachmatrix!(A::MPEM)
    c = 1.0
    for m in A
        mm = maximum(abs, m)
        if !any(isnan, mm) && !any(isinf, mm)
            m ./= mm
            c *= mm
        end
    end
    c
end

-(A::T, B::T) where {T<:MPEM2} = MPEM2([AA .- BB for (AA,BB) in zip(A.tensors,B.tensors)])
-(A::T, B::T) where {T<:MPEM3} = MPEM3([AA .- BB for (AA,BB) in zip(A.tensors,B.tensors)])

function Base.isapprox(A::T, B::T; kw...) where {T<:MPEM}
    isapprox(A.tensors, B.tensors; kw...)
end 