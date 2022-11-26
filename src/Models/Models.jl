module Models

import MatrixProductBP: exact_prob, onebpiter!, getT, getq, mpbp, 
    kron2, idx_to_value, f_bp, pair_belief_tu, onebpiter_dummy_neighbor,
    beliefs, beliefs_tu, firstvar_marginal_tu
using MatrixProductBP

import IndexedGraphs: IndexedGraph, IndexedBiDiGraph, ne, nv, outedges, idx,
    inedges, neighbors, edges, vertices
import UnPack: @unpack
import SparseArrays: nonzeros, nzrange, rowvals
import TensorCast: @reduce, @cast, TensorCast 
import ProgressMeter: Progress, next!
import LogExpFunctions: xlogx, xlogy
import Statistics: mean

export 
    SimpleBPFactor,
    Ising, Glauber, 
    q_glauber, HomogeneousGlauberFactor, GenericGlauberFactor, 
    onebpiter!, pair_observations_directed, 
    pair_observations_nondirected, mpbp, beliefs, beliefs_tu,
    iterate_bp_infinite_graph, observables_infinite_graph,
    SIS, SISFactor, q_sis, SUSCEPTIBLE, INFECTED,
    kl_marginals, l1_marginals, find_infected_bp, auc

include("simple_bp_factor.jl")

include("glauber/glauber.jl")
include("glauber/glauber_bp.jl")

include("sis/sis.jl")
include("sis/sis_bp.jl")
include("sis/sis_inference.jl")

end # end module