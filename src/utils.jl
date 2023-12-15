# an array with unspecified size and all elements equal to one
struct AllOneTensor; end
Base.getindex(::AllOneTensor, idx...) = 1
Base.size(::AllOneTensor, i::Integer) = 1
Base.axes(::AllOneTensor, i::Integer) = 1:1

# SAMPLING
# sample an index `i` of `w` with probability prop to `w[i]`
# copied from StatsBase but avoids creating a `Weight` object
function sample_noalloc(rng::AbstractRNG, w) 
    t = rand(rng)# * sum(w)
    i = 0
    cw = 0.0
    for p in w
        cw += p
        i += 1
        cw > t && return i
    end
    @assert false
end
sample_noalloc(w) = sample_noalloc(GLOBAL_RNG, w)


