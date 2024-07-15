module MatrixProductBP

using InvertedIndices: Not
using ProgressMeter: Progress, ProgressUnknown, next!
using TensorCast: @reduce, @cast, TensorCast
using LoopVectorization
using Tullio: @tullio
using IndexedGraphs: IndexedGraphs, AbstractIndexedDiGraph, IndexedGraph, IndexedBiDiGraph,
    nv, ne, edges, vertices, inedges, outedges, src, dst, idx, neighbors, IndexedEdge,
    issymmetric
using UnPack: @unpack
using Random: shuffle!, AbstractRNG, GLOBAL_RNG
using SparseArrays: rowvals, nonzeros, nzrange, sparse
using Distributions: Distributions, sample, sample!
using Measurements: Measurement, ±, value, uncertainty
using Statistics: mean, std
using Unzip: unzip
using StatsBase: weights, proportions
using LogExpFunctions: logistic, logsumexp
using .Threads: SpinLock, lock, unlock, @threads
using Lazy: @forward
using CavityTools: cavity
using LogarithmicNumbers: ULogarithmic, Logarithmic
using LinearAlgebra: I, tr
using Kronecker: kronecker

using TensorTrains:
    TensorTrains,
    getindex, iterate, firstindex, lastindex, setindex!, length, eachindex, +, -, isapprox,
    SVDTrunc, TruncBond, TruncThresh, TruncBondMax, TruncBondThresh, summary_compact,
    AbstractTensorTrain, PeriodicTensorTrain, TensorTrain, normalize_eachmatrix!,
    check_bond_dims, evaluate,
    bond_dims, flat_tt, rand_tt, flat_periodic_tt, rand_periodic_tt, 
    orthogonalize_right!, orthogonalize_left!, compress!,
    marginals, twovar_marginals, normalization, normalize!,
    svd, _compose, accumulate_L, accumulate_R

using TensorTrains.UniformTensorTrains:
    InfiniteUniformTensorTrain, flat_infinite_uniform_tt


export 
    SVDTrunc, TruncBond, TruncThresh, TruncBondMax, TruncBondThresh,
    PeriodicMPEM2, PeriodicMPEM3, PeriodicMPEM1,
    MPEM1, MPEM2, MPEM3, mpem2, rand_mpem1, rand_mpem2, 
    InfiniteUniformMPEM1, InfiniteUniformMPEM2, InfiniteUniformMPEM3,
    normalization, normalize!, marginalize,
    orthogonalize_right!, orthogonalize_left!, compress!, twovar_marginals, evaluate,
    BPFactor, nstates, MPBP, mpbp, reset_messages!, reset_beliefs!, reset_observations!,
    reset!, is_free_dynamics, onebpiter!, CB_BP, iterate!, 
    pair_beliefs, pair_beliefs_as_mpem, beliefs_tu, autocorrelations,
    autocovariances, means, beliefs, bethe_free_energy, 
    periodic_mpbp, is_periodic,
    mpbp_infinite_graph, InfiniteRegularGraph, periodic_mpbp_infinite_graph,
    InfiniteBipartiteRegularGraph, mpbp_infinite_bipartite_graph,
    logprob, expectation, pair_observations_directed, 
    pair_observations_nondirected, pair_obs_undirected_to_directed,
    exact_prob, exact_marginals, site_marginals, exact_autocorrelations,
    exact_autocovariances, exact_marginal_expectations, 
    SoftMarginSampler, onesample!, onesample, sample!, sample, marginals, pair_marginals,
    continuous_sis_sampler, simulate_queue_sis!,
    draw_node_observations!, AtomicVector,
    RecursiveBPFactor, DampedFactor, RecursiveTraceFactor, GenericFactor,
    RestrictedRecursiveBPFactor,
    mpbp_stationary, mpbp_stationary_infinite_graph, mpbp_stationary_infinite_bipartite_graph,
    mean_with_uncertainty


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
include("stationary.jl")

include("Models/Models.jl")

end # end module