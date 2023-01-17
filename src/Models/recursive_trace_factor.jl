struct RecursiveTraceFactor{F<:BPFactor,N} <: RecursiveBPFactor
    w :: F
end
RecursiveTraceFactor(f,N) = RecursiveTraceFactor{typeof(f),N}(f)

nstates(::Type{RecursiveTraceFactor{F,N}}, d::Integer) where {F,N} = N^d

function prob_y(wᵢ::U, xᵢᵗ⁺¹, xᵢᵗ, yₙᵢᵗ, dᵢ) where {U <: RecursiveTraceFactor{F,N}} where {F,N}
    #@show yₙᵢᵗ
    #@show reverse(digits(yₙᵢᵗ-1; base=N, pad=dᵢ)) .+ 1
    #@show wᵢ.w(xᵢᵗ⁺¹, reverse(digits(yₙᵢᵗ-1; base=N, pad=dᵢ)) .+ 1, xᵢᵗ)
    wᵢ.w(xᵢᵗ⁺¹, reverse(digits(yₙᵢᵗ-1; base=N, pad=dᵢ)) .+ 1, xᵢᵗ)
end

prob_xy(wᵢ::RecursiveTraceFactor, yₖ, xₖ, xᵢ, k) = (yₖ == xₖ)

prob_yy(wᵢ::U, y, y1, y2, xᵢ, d1, d2) where {U<:RecursiveTraceFactor} = (y - 1 == (y1 - 1) * d2 + (y2 - 1))
