# Matrix Product Belief Propagation

When possible, variable names match the notation used in the [notes](https://www.overleaf.com/read/cjtftmgvyxkt).

This folder is organized as follows:
- A self-contained submodule `MPEMs` (Matrix Product Edge Messages) to operate with matrix products. 
There are two types of `MPEM's:
    - `MPEM2`: product of matrices depending on two variables (e.g. $x_i^t$, $x_j^t$)
    - `MPEM3`: product of matrices depending on three variables (e.g. $x_i^{t+1}$, $x_i^t$, $x_j^t$)
    
    Some operations on MPEMs:
    - `evaluate(A::MPEM, x)` the matrix product on a specific trajectory $\\{(x_i^t,x_j^t)\\}_{t=0:T}$ to get its probability
    - `normalize!(A::MPEM2)` the matrix product to sum to 1, return the computed normalization (useful for computing the free energy)
    - `sweep_RtoL!(C::MPEM2; svd_trunc::SVDTrunc)`, `sweep_RtoL!(C::MPEM2; svd_trunc::SVDTrunc)`: perform a SVD sweep and optionally truncate. For no truncation, set optional argument `svd_trunc=TruncTresh(0.0)`
    - `SVDTrunc` can be picked to be a `TruncThresh` (truncate according to a threshold), or a `TruncBond` (truncate to maximum bond size)
    - `mpem2(B::MPEM3)`: convert a `MPEM3` into a `MPEM2` via SVDs (as in section 3.1 in the notes)
    - `bond_dims`: display the size of the matrices in the product

- The main module `MatrixProductBP` implementing the BP equations.
    Essential ingredients are:
    - An `IndexedBiDiGraph` on which the system is defined
    - `BPFactor`s living on the edges of the graph. Any concrete subtype (e.g. `SISFactor`) must implement a method `(fᵢ::SISFactor)(xᵢᵗ⁺¹::Integer, xₙᵢᵗ::Vector{<:Integer}, xᵢᵗ::Integer)` that computes the contribution of that factor to the joint probability as a function of $x_i^{t+1},\boldsymbol{x}_{\partial i}^{t}, x_i^{t}$.
    - Single-node observations `ϕ` and edge observations `ψ`
    - Time zero prior probability `p⁰`
    
    Some allowed operations:
    - `iterate!(bp::MPBP; kw...)`
    - `pair_beliefs(bp::MPBP)`
    - `beliefs(bp::MPBP)`
    - `bethe_free_energy(bp::MPBP)`
    
    The is some montecarlo (Soft Margin) utility in `src/sampling/`:
    - `sample(bp::MPBP, nsamples::Integer)`: draw samples from the prior weighted with their likelihood
    - `draw_node_observations!(bp::MPBP, nobs::Integer)`: draw samples from the prior and add them to the `MPBP` object as observations
    
    Code for computing probabilities exacyly by exhaustive enumeration in `src/exact/` (only for small system size and final time):
    - `exact_prob(bp::MPBP)`
    
- A submodule `Models` that can be included with `using MatrixProductBP.Models` that uses the framework for Glauber dynamics and the SIS model of epidemic spreading.
It contains model definitions and specialized versions of the BP equations that are more efficient than the generic one for large degree.

- A `test` folder with small examples 

- A `notebooks` folder showing the applications of the method. These notebooks should, ideally, reproduce the results in the future article.
 
## TO DO:
- normalize each matrix
- log of partial free energies
- reasoning on matrix size for SI
- compare magnetizations instead of KLs for binary variables
