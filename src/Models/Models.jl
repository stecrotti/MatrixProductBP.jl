module Models

import MatrixProductBP: exact_prob, getT, nstates, mpbp, compress!,
    f_bp, f_bp_dummy_neighbor, onebpiter_dummy_neighbor,
    beliefs, beliefs_tu, marginals, pair_belief, pair_beliefs,
    marginalize, cavity, onebpiter!, check_ψs, _compose,
    RecursiveBPFactor, DampedFactor, nstates, prob_y, prob_xy, prob_yy, prob_y0, prob_y_partial,
    prob_y_dummy, periodic_mpbp, AbstractMPEM2, MPEM2, MPEM3, mpem2, f_bp_partial_i, f_bp_partial_ij, compute_prob_ys, mpbp_stationary
using MatrixProductBP

import IndexedGraphs: IndexedGraph, IndexedDiGraph, IndexedBiDiGraph, AbstractIndexedDiGraph, ne, nv, 
    outedges, idx, src, dst, inedges, neighbors, edges, vertices, IndexedEdge
import UnPack: @unpack
import SparseArrays: nonzeros, nzrange, rowvals, Symmetric, SparseMatrixCSC, sparse
import TensorCast: @reduce, @cast, TensorCast 
import ProgressMeter: Progress, next!, ProgressUnknown
import LogExpFunctions: xlogx, xlogy
import Statistics: mean, std
import Measurements: Measurement, ±
import LoopVectorization
import Tullio: @tullio
import Unzip: unzip
import Distributions: rand, Poisson, Distribution, Dirac, MixtureModel
import Random: GLOBAL_RNG, shuffle!
import Lazy: @forward
import LogarithmicNumbers: ULogarithmic, Logarithmic
import LinearAlgebra: issymmetric
# import HypergeometricFunctions: _₂F₁
using Nemo
using Nemo: hypergeometric_2f1, AcbField, AcbFieldElem
import TensorTrains: getindex, iterate, firstindex, lastindex, setindex!, length, eachindex, +, -,
    SVDTrunc, TruncBond, TruncThresh, TruncBondMax, TruncBondThresh,
    AbstractTensorTrain, PeriodicTensorTrain, TensorTrain, normalize_eachmatrix!,
    check_bond_dims, evaluate,
    bond_dims, flat_tt, rand_tt,
    orthogonalize_right!, orthogonalize_left!, compress!,
    marginals, twovar_marginals, normalization, normalize!,
    svd, _compose, accumulate_L, accumulate_R
import OffsetArrays: OffsetArray, OffsetVector    

export
    Ising, Glauber, energy,
    HomogeneousGlauberFactor, GenericGlauberFactor, DampedGlauberFactor, PMJGlauberFactor, mpbp,
    equilibrium_magnetization, equilibrium_observables, RandomRegular, ErdosRenyi, CB_Pop,
    SIS, SISFactor, SIRS, SIRSFactor, SIS_heterogeneous, SIS_heterogeneousFactor, SUSCEPTIBLE, INFECTIOUS, RECOVERED,
    kl_marginals, l1_marginals, roc, auc,
    potts2spin, spin2potts,
    fourier_tensor_train, flat_fourier_tt, rand_fourier_tt, fourier_tensor_train_spin, marginals_fourier,
    FourierBPFactor, FourierGlauberFactor, mpbp_fourier, f_bp_partial_i, f_bp_partial_ij, compute_prob_ys

include("glauber_integer/glauber.jl")
include("glauber_integer/glauber_bp.jl")
include("glauber_integer/equilibrium.jl")

include("glauber_fourier/glauber_fourier.jl")
include("glauber_fourier/fourier_tensor_train.jl")
include("glauber_fourier/bp_fourier.jl")


include("epidemics/sis.jl")
include("epidemics/sis_bp.jl")
include("epidemics/inference.jl")
include("epidemics/sirs.jl")
include("epidemics/sirs_bp.jl")
include("epidemics/sis_heterogeneous.jl")
include("epidemics/sis_heterogeneous_bp.jl")

end # end module
