MPEM1(tensors::Vector{Array{Float64,3}}) = MatrixProductTrain(tensors)

# can evaluate a MPEM1 with a vector of integers instead of a vector whose elements are 
#  1-element vectors of integers as expected by the MatrixProductTrain interface
function evaluate(A::MPEM1, x::Vector{U}) where {U<:Integer}
    only(prod(@view a[:, :, xx...] for (a,xx) in zip(A, x)))
end

# construct a uniform mpem with given bond dimensions
mpem1(q::Int, T::Int; d::Int=2, bondsizes=[1; fill(d, T); 1]) = mpem(bondsizes, q)

# construct a uniform mpem with given bond dimensions
rand_mpem1(q::Int, T::Int; d::Int=2, bondsizes=[1; fill(d, T); 1]) = rand_mpem(bondsizes, q)

nstates(A::MPEM1) = size(A[1],3)