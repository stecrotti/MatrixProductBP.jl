import Random


# maps (1,2) -> (1,-1)
potts2spin(x) = 3-2x
spin2potts(σ) = (3+σ)/2

struct AllOneTensor; end
Base.getindex(::AllOneTensor, idx...) = 1
Base.size(::AllOneTensor, i::Integer) = 1
Base.axes(::AllOneTensor, i::Integer) = 1:1

# symmetrize and set diagonal to zero
function symmetrize_nodiag!(A::AbstractMatrix)
    A .= (A+A')/2
    for i in axes(A, 1)
        A[i,i] = 0
    end
    A
end

# SAMPLING
# sample an index `i` of `w` with probability prop to `w[i]`
# copied from StatsBase but avoid creating a `Weight` object
function sample_noalloc(rng::Random.AbstractRNG, w::AbstractVector) 
    t = rand(rng) * sum(w)
    n = length(w)
    i = 1
    cw = w[1]
    while cw < t && i < n
        i += 1
        @inbounds cw += w[i]
    end
    return i
end
sample_noalloc(w::AbstractVector) = sample_noalloc(Random.GLOBAL_RNG, w)