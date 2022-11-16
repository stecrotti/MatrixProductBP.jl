module MPEMs

using Reexport
@reexport import Base:
    -, eltype, getindex, iterate, firstindex, lastindex, setindex!, length,
    isapprox
import Lazy: @forward
import TensorCast: @cast, @reduce, TensorCast
import LinearAlgebra: svd, norm, normalize!
import Tullio: @tullio
import Random: AbstractRNG, GLOBAL_RNG

export 
    SVDTrunc, TruncBond, TruncThresh,
    MPEM, normalize_eachmatrix!, -, isapprox, evaluate, getq, getT, bond_dims,
    MPEM2, mpem2, rand_mpem2, sweep_RtoL!, sweep_LtoR!,
    normalization, normalize!,
    MPEM3

# Matrix Product Edge Message
abstract type MPEM; end

include("svd_trunc.jl")
include("mpem2.jl")
include("mpem3.jl")


# keep size of matrix elements under control by dividing by the max
# return the log of the product of the individual normalizations 
function normalize_eachmatrix!(A::MPEM)
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

-(A::T, B::T) where {T<:MPEM2} = MPEM2([AA .- BB for (AA,BB) in zip(A.tensors,B.tensors)])
-(A::T, B::T) where {T<:MPEM3} = MPEM3([AA .- BB for (AA,BB) in zip(A.tensors,B.tensors)])

function isapprox(A::T, B::T; kw...) where {T<:MPEM}
    isapprox(A.tensors, B.tensors; kw...)
end 

end # end module