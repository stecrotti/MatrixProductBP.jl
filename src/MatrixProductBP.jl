module MatrixProductBP

import InvertedIndices: Not
import ProgressMeter: Progress, ProgressUnknown, next!
import TensorCast: @reduce, @cast, TensorCast 
import Tullio: @tullio
import IndexedGraphs: nv, ne, edges, vertices, IndexedBiDiGraph,
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
    BPFactor, MPBP, mpbp, reset_messages!, onebpiter!, CB_BP, iterate!, pair_beliefs,
    beliefs, bethe_free_energy,
    exact_prob, site_marginals, site_time_marginals, 
    SoftMarginSampler, onesample!, onesample, sample!, sample, marginals, 
    draw_node_observations!


include("utils.jl")
include("MPEMs.jl")
using .MPEMs, Reexport
@reexport import .MPEMs: SVDTrunc, TruncBond, TruncThresh,
    MPEM, normalize_eachmatrix!, -, isapprox, evaluate, getq, getT, bond_dims,
    MPEM2, mpem2, rand_mpem2, sweep_RtoL!, sweep_LtoR!,
    normalization, normalize!, MPEM3

include("bp.jl")
include("mpbp.jl")
include("exact.jl")
include("sampling.jl")

include("Models/Models.jl")
using .Models
import .Models: Ising, Glauber, exact_prob,
    q_glauber, GlauberFactor, onebpiter!, pair_observations_directed, 
    pair_observations_nondirected, magnetizations, mpbp,
    SIS, q_sis, SUSCEPTIBLE, INFECTED

end # end module