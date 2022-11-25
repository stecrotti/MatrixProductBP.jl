module Models

import MatrixProductBP: exact_prob, _onebpiter!, onebpiter!, getT, getq, mpbp, 
    kron2, idx_to_value, pair_belief_tu, onebpiter_dummy_neighbor,
    _onebpiter_dummy_neighbor, beliefs, beliefs_tu, firstvar_marginal_tu
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
    Ising, Glauber, 
    q_glauber, GlauberFactor, HomogeneousGlauberFactor, GenericGlauberFactor, 
    onebpiter!, pair_observations_directed, 
    pair_observations_nondirected, magnetizations, mpbp, beliefs, beliefs_tu,
    glauber_infinite_graph, observables_glauber_infinite_graph,
    SIS, SISFactor, q_sis, SUSCEPTIBLE, INFECTED,
    kl_marginals, l1_marginals, find_infected_bp, auc

include("glauber/glauber.jl")
include("glauber/glauber_bp.jl")
include("glauber/glauber_infinite_graph.jl")
include("sis/sis.jl")
include("sis/sis_bp.jl")
include("sis/sis_inference.jl")

end # end module