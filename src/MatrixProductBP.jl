module MatrixProductBP

import InvertedIndices: Not
import ProgressMeter: Progress, ProgressUnknown, next!
import TensorCast: @reduce, @cast, TensorCast 
import Tullio: @tullio
import IndexedGraphs: nv, ne, edges, vertices, IndexedBiDiGraph, IndexedGraph,
    inedges, outedges, src, dst, idx, neighbors
import UnPack: @unpack
import Random: shuffle!, AbstractRNG, GLOBAL_RNG
import SparseArrays: rowvals, nonzeros, nzrange
import Distributions: sample, sample!, Bernoulli
import Measurements: Measurement, ±
import Statistics: mean, std
import Unzip: unzip
import StatsBase: weights, proportions
import LogExpFunctions: logistic, logsumexp

export
    BPFactor, MPBP, mpbp, reset_messages!, onebpiter!, CB_BP, iterate!, 
    pair_beliefs, pair_beliefs_tu, beliefs_tu, autocorrelations,
    autocovariances, beliefs, bethe_free_energy, 
    logprob, marginal_to_expectation,
    exact_prob, exact_marginals, site_marginals, exact_autocorrelations,
    exact_autocovariances, exact_marginal_expectations, 
    SoftMarginSampler, onesample!, onesample, sample!, sample, marginals, 
    draw_node_observations!


include("utils.jl")
include("MPEMs/MPEMs.jl")
using .MPEMs, Reexport
@reexport import .MPEMs: SVDTrunc, TruncBond, TruncThresh,
    MPEM, normalize_eachmatrix!, -, isapprox, evaluate, getq, getT, bond_dims,
    MPEM2, mpem2, rand_mpem2, sweep_RtoL!, sweep_LtoR!,
    accumulate_L, accumulate_R, accumulate_M, pair_marginal, firstvar_marginal,
    pair_marginal_tu, firstvar_marginal_tu,
    normalization, normalize!, MPEM3

include("bp_core.jl")
include("mpbp.jl")
include("exact.jl")
include("sampling.jl")

include("Models/Models.jl")

end # end module