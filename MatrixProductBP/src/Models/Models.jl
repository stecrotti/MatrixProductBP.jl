module Models

import MatrixProductBP: exact_prob, onebpiter!, getT, getq, mpbp
using MatrixProductBP

import IndexedGraphs: IndexedGraph, IndexedBiDiGraph, ne, nv, outedges, idx
import UnPack: @unpack

export 
    Ising, Glauber, exact_prob, site_marginals, site_time_marginals, 
    site_time_magnetizations, 
    q_glauber, GlauberFactor, onebpiter!, pair_observations_directed, 
    pair_observations_nondirected, magnetizations, mpbp,
    SIS, q_sis, SUSCEPTIBLE, INFECTED

include("glauber/glauber.jl")
include("glauber/bp_glauber.jl")
include("sis/sis.jl")
include("sis/bp_sis.jl")

end # end module