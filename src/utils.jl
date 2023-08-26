# an array with unspecified size and all elements equal to one
struct AllOneTensor; end
Base.getindex(::AllOneTensor, idx...) = 1
Base.size(::AllOneTensor, i::Integer) = 1
Base.axes(::AllOneTensor, i::Integer) = 1:1

# compute the kronecker product only over needed indices
kron2() = AllOneTensor()
function kron2(A₁::Array{F,4}) where F
    @cast _[m₁, n₁, xᵢ, x₁] := A₁[m₁, n₁, x₁, xᵢ]
end
function kron2(A₁::Array{F,4}, A₂::Array{F,4}) where F
    @cast _[(m₁, m₂), (n₁, n₂), xᵢ, x₁, x₂] := A₁[m₁, n₁, x₁, xᵢ] * 
        A₂[m₂, n₂, x₂, xᵢ]
end
function kron2(A₁::Array{F,4}, A₂::Array{F,4}, A₃::Array{F,4}) where F
    @cast _[(m₁, m₂, m₃), (n₁, n₂, n₃), xᵢ, x₁, x₂, x₃] := 
        A₁[m₁, n₁, x₁, xᵢ] * A₂[m₂, n₂, x₂, xᵢ] * A₃[m₃, n₃, x₃, xᵢ]
end
function kron2(A₁::Array{F,4}, A₂::Array{F,4}, A₃::Array{F,4}, A₄::Array{F,4}) where F
    @cast _[(m₁, m₂, m₃, m₄), (n₁, n₂, n₃, n₄), xᵢ, x₁, x₂, x₃, x₄] := 
        A₁[m₁, n₁, x₁, xᵢ] * A₂[m₂, n₂, x₂, xᵢ] * A₃[m₃, n₃, x₃, xᵢ] * A₄[m₄, n₄, x₄, xᵢ]
end
function kron2(A₁::Array{F,4}, A₂::Array{F,4}, A₃::Array{F,4}, A₄::Array{F,4}, 
        A₅::Array{F,4}) where F
    @cast _[(m₁, m₂, m₃, m₄, m₅), (n₁, n₂, n₃, n₄, n₅), xᵢ, x₁, x₂, x₃, x₄, x₅] := 
        A₁[m₁, n₁, x₁, xᵢ] * A₂[m₂, n₂, x₂, xᵢ] * A₃[m₃, n₃, x₃, xᵢ] * A₄[m₄, n₄, x₄, xᵢ] *
            A₅[m₅, n₅, x₅, xᵢ]
end
function kron2(A₁::Array{F,4}, A₂::Array{F,4}, A₃::Array{F,4}, A₄::Array{F,4}, 
    A₅::Array{F,4}, A₆::Array{F,4}) where F
@cast _[(m₁, m₂, m₃, m₄, m₅, m₆), (n₁, n₂, n₃, n₄, n₅, n₆), xᵢ, x₁, x₂, x₃, x₄, x₅, x₆] := 
    A₁[m₁, n₁, x₁, xᵢ] * A₂[m₂, n₂, x₂, xᵢ] * A₃[m₃, n₃, x₃, xᵢ] * A₄[m₄, n₄, x₄, xᵢ] *
        A₅[m₅, n₅, x₅, xᵢ] * A₆[m₆, n₆, x₆, xᵢ]
end

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


