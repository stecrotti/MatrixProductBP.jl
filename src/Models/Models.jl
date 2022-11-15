module Models

import MatrixProductBP: exact_prob, onebpiter!, getT, getq, mpbp, kron2
using MatrixProductBP

import IndexedGraphs: IndexedGraph, IndexedBiDiGraph, ne, nv, outedges, idx,
    inedges, neighbors, edges, vertices
import UnPack: @unpack
import SparseArrays: nonzeros, nzrange, rowvals
import TensorCast: @reduce, @cast, TensorCast 
import ProgressMeter: Progress, next!
import LogExpFunctions: xlogx, xlogy

export 
    Ising, Glauber, exact_prob, 
    site_time_magnetizations, 
    q_glauber, GlauberFactor, HomogeneousGlauberFactor, GenericGlauberFactor, 
    onebpiter!, pair_observations_directed, 
    pair_observations_nondirected, magnetizations, mpbp,
    SIS, q_sis, SUSCEPTIBLE, INFECTED,
    kl_marginals, find_infected_bp, auc

include("glauber/glauber.jl")
include("glauber/glauber_bp.jl")
include("sis/sis.jl")
include("sis/sis_bp.jl")
include("sis/sis_inference.jl")

end # end module