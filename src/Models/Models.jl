module Models



import MatrixProductBP: exact_prob, getT, nstates, mpbp, 
    kron2, f_bp, f_bp_dummy_neighbor, onebpiter_dummy_neighbor,
    beliefs, beliefs_tu, marginals, 
    marginalize, cavity, onebpiter!, check_ψs
using MatrixProductBP

import IndexedGraphs: IndexedGraph, IndexedBiDiGraph, AbstractIndexedDiGraph, ne, nv, 
    outedges, idx, src, dst, inedges, neighbors, edges, vertices, IndexedEdge
import UnPack: @unpack
import SparseArrays: nonzeros, nzrange, rowvals
import TensorCast: @reduce, @cast, TensorCast 
import ProgressMeter: Progress, next!
import LogExpFunctions: xlogx, xlogy
import Statistics: mean, std
import Measurements: Measurement, ±
import Tullio: @tullio
import Unzip: unzip
import Distributions: rand, Poisson, truncated

export 
    RecursiveBPFactor, beliefs, beliefs_tu,
    mpbp_infinite_graph,
    Ising, Glauber, 
    HomogeneousGlauberFactor, GenericGlauberFactor, PMJGlauberFactor, mpbp,
    equilibrium_magnetization, RandomRegular, ErdosRenyi,
    SIS, SISFactor, SIRS, SIRSFactor, SUSCEPTIBLE, INFECTED, RECOVERED,
    kl_marginals, l1_marginals, roc, auc,
    RecursiveTraceFactor, GenericFactor, RestrictedRecursiveBPFactor


include("recursive_bp_factor.jl")
include("test_factors.jl")
include("infinite_graph.jl")

include("glauber/glauber.jl")
include("glauber/glauber_bp.jl")
include("glauber/equilibrium.jl")

include("epidemics/sis.jl")
include("epidemics/sis_bp.jl")
include("epidemics/inference.jl")
include("epidemics/sirs.jl")
include("epidemics/sirs_bp.jl")

end # end module
