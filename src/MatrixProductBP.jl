module MatrixProductBP

import InvertedIndices: Not
import ProgressMeter: Progress, ProgressUnknown, next!
import TensorCast: @reduce, @cast, TensorCast 
import LoopVectorization
import Tullio: @tullio
import IndexedGraphs: nv, ne, edges, vertices, AbstractIndexedDiGraph, IndexedGraph,
    IndexedBiDiGraph, inedges, outedges, src, dst, idx, neighbors
import UnPack: @unpack
import Random: shuffle!, AbstractRNG, GLOBAL_RNG
import SparseArrays: rowvals, nonzeros, nzrange
import Distributions: sample, sample!, Bernoulli
import Measurements: Measurement, ±
import Statistics: mean, std
import Unzip: unzip
import StatsBase: weights, proportions
import LogExpFunctions: logistic, logsumexp
import .Threads: SpinLock, lock, unlock
import Lazy: @forward
import CavityTools: cavity


export
    BPFactor, nstates, MPBP, mpbp, reset_messages!, reset_beliefs!, reset_observations!,
    reset!, onebpiter!, CB_BP, iterate!, 
    pair_beliefs, pair_beliefs_tu, beliefs_tu, autocorrelations,
    autocovariances, means, beliefs, bethe_free_energy, 
    logprob, expectation, pair_observations_directed, 
    pair_observations_nondirected, pair_obs_undirected_to_directed,
    exact_prob, exact_marginals, site_marginals, exact_autocorrelations,
    exact_autocovariances, exact_marginal_expectations, 
    SoftMarginSampler, onesample!, onesample, sample!, sample, marginals, pair_marginals,
    continuous_sis_sampler, simulate_queue_sis!,
    draw_node_observations!, AtomicVector


include("utils.jl")
include("atomic_vector.jl")
include("MPEMs/MPEMs.jl")
using .MPEMs, Reexport
@reexport import .MPEMs: SVDTrunc, TruncBond, TruncThresh, TruncBondMax, TruncBondThresh, 
    summary_compact, normalize_eachmatrix!, -, isapprox, evaluate, getT, bond_dims,
    MPEM, MPEM2, MPEM3, MatrixProductTrain, mpem2, rand_mpem2, sweep_RtoL!, sweep_LtoR!,
    compress!, accumulate_L, accumulate_R, accumulate_M, pair_marginal, firstvar_marginal,
    pair_marginal_tu, firstvar_marginal_tu, marginals, marginals_tu, mpem1,
    normalization, normalize!,  nstates, marginalize

include("bp_core.jl")
include("mpbp.jl")
include("exact.jl")
include("sampling.jl")

include("Models/Models.jl")

end # end module