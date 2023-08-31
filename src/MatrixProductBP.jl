module MatrixProductBP

import InvertedIndices: Not
import ProgressMeter: Progress, ProgressUnknown, next!
import TensorCast: @reduce, @cast, TensorCast 
import LoopVectorization
import Tullio: @tullio
import IndexedGraphs: nv, ne, edges, vertices, AbstractIndexedDiGraph, IndexedGraph,
    IndexedBiDiGraph, inedges, outedges, src, dst, idx, neighbors, IndexedEdge
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
import LogarithmicNumbers: ULogarithmic

import TensorTrains:
    getindex, iterate, firstindex, lastindex, setindex!, length, eachindex, +, -, isapprox,
    SVDTrunc, TruncBond, TruncThresh, TruncBondMax, TruncBondThresh, summary_compact,
    TensorTrain, normalize_eachmatrix!, check_bond_dims, evaluate,
    bond_dims, uniform_tt, rand_tt, orthogonalize_right!, orthogonalize_left!, compress!,
    marginals, twovar_marginals, normalization, normalize!,
    svd, _compose, accumulate_L, accumulate_R


export 
    SVDTrunc, TruncBond, TruncThresh, TruncBondMax, TruncBondThresh,
    MPEM1, MPEM2, MPEM3, mpem2, normalization, normalize!, orthogonalize_right!, 
    orthogonalize_left!, compress!, twovar_marginals, evaluate,
    BPFactor, nstates, MPBP, mpbp, reset_messages!, reset_beliefs!, reset_observations!,
    reset!, is_free_dynamics, onebpiter!, CB_BP, iterate!, 
    pair_beliefs, pair_beliefs_as_mpem, pair_beliefs_tu, beliefs_tu, autocorrelations,
    autocovariances, means, beliefs, bethe_free_energy, 
    mpbp_infinite_graph, InfiniteRegularGraph,
    logprob, expectation, pair_observations_directed, 
    pair_observations_nondirected, pair_obs_undirected_to_directed,
    exact_prob, exact_marginals, site_marginals, exact_autocorrelations,
    exact_autocovariances, exact_marginal_expectations, 
    SoftMarginSampler, onesample!, onesample, sample!, sample, marginals, pair_marginals,
    continuous_sis_sampler, simulate_queue_sis!,
    draw_node_observations!, AtomicVector,
    RecursiveBPFactor, DampedFactor, RecursiveTraceFactor, GenericFactor,
    RestrictedRecursiveBPFactor


include("utils.jl")
include("atomic_vector.jl")
include("mpems.jl")
include("bp_core.jl")
include("mpbp.jl")
include("recursive_bp_factor.jl")
include("test_factors.jl")
include("infinite_graph.jl")
include("exact.jl")
include("sampling.jl")

include("Models/Models.jl")

end # end module