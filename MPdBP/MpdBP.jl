module MPdBP

import InvertedIndices: Not
import ProgressMeter: Progress, next!
import TensorCast: @reduce
import Tullio: @tullio
import IndexedGraphs: nv, ne, edges, vertices, IndexedBiDiGraph,
    inedges, outedges, src, dst, idx
import UnPack: @unpack
import Random: shuffle!, AbstractRNG, GLOBAL_RNG
import SparseArrays: rowvals, nonzeros, nzrange

export
    MPdBP, mpdbp, reset_messages!, onebpiter!, CB_BP, iterate!, pair_beliefs,
    beliefs, bethe_free_energy


include("utils.jl")
include("MPEMs.jl")
using .MPEMs
include("bp.jl")
include("mpdbp.jl")


end # end module