module MPEMs

using Reexport
@reexport import Base:
    eltype, getindex, iterate, firstindex, lastindex, setindex!, eachindex, 
    length, isapprox, +, -, show
import Lazy: @forward
import TensorCast: @cast, @reduce, TensorCast
import LinearAlgebra: svd, norm, normalize!
import Tullio: @tullio
import Random: AbstractRNG, GLOBAL_RNG
import StatsBase: sample!, sample

export 
    SVDTrunc, TruncBond, TruncThresh, TruncBondMax, TruncBondThresh, summary_compact,
    MPEM, normalize_eachmatrix!, +, -, isapprox, evaluate, getT, bond_dims,
    MPEM2, mpem2, rand_mpem2, sweep_RtoL!, sweep_LtoR!, compress!,
    MPEM1, mpem1, marginalize, marginals, 
    accumulate_L, accumulate_R, accumulate_M, firstvar_marginal,
    marginals_tu, normalization, normalize!,
    MPEM3, MatrixProductTrain, nstates,
    sample!, sample

# Matrix Product Edge Message

abstract type MPEM end

include("../utils.jl")
include("svd_trunc.jl")
include("mptrain.jl")
include("mpem1.jl")
include("mpem2.jl")
include("mpem3.jl")



end # end module
