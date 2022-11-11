module MatrixProductBP

import InvertedIndices: Not
import ProgressMeter: Progress, next!
import TensorCast: @reduce
import Tullio: @tullio
import IndexedGraphs: nv, ne, edges, vertices, IndexedBiDiGraph,
    inedges, outedges, src, dst, idx
import UnPack: @unpack
import Random: shuffle!, AbstractRNG, GLOBAL_RNG
import SparseArrays: rowvals, nonzeros, nzrange
import Distributions: sample!, sample, Bernoulli
import Measurements: Measurement
import Statistics: mean, std
import Unzip: unzip
import StatsBase: weights, proportions
import LogExpFunctions: logistic

export
    MPBP, mpbp, reset_messages!, onebpiter!, CB_BP, iterate!, pair_beliefs,
    beliefs, bethe_free_energy,
    exact_prob,
    SoftMarginSampler, onesample!, onesample, sample, marginals, 
    draw_node_observations!, 


include("utils.jl")
include("MPEMs.jl")
using .MPEMs
include("bp.jl")
include("mpbp.jl")
include("exact.jl")
include("sampling.jl")
include("Models/Models.jl")


end # end module