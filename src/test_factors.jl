"""
`RecursiveTraceFactor` transforms any `<:BPFactor`` in a `<:RecursiveBPFactor`. Works only with 
homogeneous variables with `N` states
"""
struct RecursiveTraceFactor{F<:BPFactor,N} <: RecursiveBPFactor
    w :: F
end

RecursiveTraceFactor(f,N) = RecursiveTraceFactor{typeof(f),N}(f)

nstates(::RecursiveTraceFactor{F,N}, d::Integer) where {F,N} = N^d

function prob_y(wᵢ::U, xᵢᵗ⁺¹, xᵢᵗ, yₙᵢᵗ, dᵢ) where {U <: RecursiveTraceFactor{F,N}} where {F,N}
    wᵢ.w(xᵢᵗ⁺¹, reverse(digits(yₙᵢᵗ-1; base=N, pad=dᵢ)) .+ 1, xᵢᵗ)
end

prob_xy(wᵢ::RecursiveTraceFactor, yₖ, xₖ, xᵢ, k) = (yₖ == xₖ)

prob_yy(wᵢ::RecursiveTraceFactor, y, y1, y2, xᵢ, d1, d2) = (y - 1 == (y1 - 1)  + (y2 - 1) * nstates(wᵢ,d1))


"""
`RestrictedRecursiveBPFactor` exercises the generic implementations of `prob_y_partial` and the energy calculation 
 (i.e. `(w::RecursiveBPFactor)(x...)``)
"""
struct RestrictedRecursiveBPFactor{F} <: RecursiveBPFactor
    w::F
end

prob_xy(w::RestrictedRecursiveBPFactor, x...) = prob_xy(w.w, x...)
prob_yy(w::RestrictedRecursiveBPFactor, x...) = prob_yy(w.w, x...)
prob_y(w::RestrictedRecursiveBPFactor, x...) = prob_y(w.w, x...)

nstates(w::RestrictedRecursiveBPFactor{F}, l::Int) where F = nstates(w.w, l)


"""
`GenericFactor{F}` transforms a specialized factor (such as `<:RecursiveBPFactor`) into a generic one which uses the
exhaustive trace for the computation of the BP update
"""
struct GenericFactor{F} <: BPFactor
    w::F
end

(w::GenericFactor)(xᵢᵗ⁺¹::Integer, xₙᵢᵗ::AbstractVector{<:Integer}, xᵢᵗ::Integer) = w.w(xᵢᵗ⁺¹, xₙᵢᵗ, xᵢᵗ)
