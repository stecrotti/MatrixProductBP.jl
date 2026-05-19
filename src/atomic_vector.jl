struct AtomicVector{T} <: AbstractVector{T}
    v::Vector{T}
    s::Threads.SpinLock
    AtomicVector(v::Vector{T}) where T = new{T}(v, SpinLock())
end

Base.convert(::Type{AtomicVector{T}}, v::Vector{T}) where T = AtomicVector(v)

@forward AtomicVector.v Base.length, Base.iterate, Base.size

function Base.getindex(a::AtomicVector, i::Int)
    lock(a.s)
    x = a.v[i]
    unlock(a.s)
    x
end

function Base.setindex!(a::AtomicVector, x, i::Int)
    lock(a.s)
    a.v[i] = x
    unlock(a.s)
    x
end

function Base.resize!(a::AtomicVector, n::Int)
    lock(a.s)
    resize!(a.v, n)
    unlock(a.s)
    a
end
