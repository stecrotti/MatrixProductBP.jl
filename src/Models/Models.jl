module Models

import MatrixProductBP: exact_prob, getT, getq, mpbp, 
    kron2, idx_to_value, f_bp, f_bp_dummy_neighbor, onebpiter_dummy_neighbor,
    pair_belief_tu, beliefs, beliefs_tu, firstvar_marginal_tu
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
    SimpleBPFactor, beliefs, beliefs_tu,
    iterate_bp_infinite_graph, observables_infinite_graph,
    Ising, Glauber, 
    q_glauber, HomogeneousGlauberFactor, GenericGlauberFactor, 
    glauber_infinite_graph,
    pair_observations_directed, 
    pair_observations_nondirected, mpbp, 
    SIS, SISFactor, q_sis, SUSCEPTIBLE, INFECTED, sis_infinite_graph,
    kl_marginals, l1_marginals, find_infected_bp, auc

include("simple_bp_factor.jl")

include("glauber/glauber.jl")
include("glauber/glauber_bp.jl")

include("sis/sis.jl")
include("sis/sis_bp.jl")
include("sis/sis_inference.jl")

end # end module