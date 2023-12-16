"""
Factor for the factor graph of a model solvable with MPBP.

Any `BPFactor` subtype must implement:
- A functor that computes the Boltzmann contribution to the joint probability

That's it!

"""
abstract type BPFactor; end

# Factors are not collections (avoid confusion in TensorCast)
Base.broadcastable(b::BPFactor) = Ref(b)

# compute outgoing message as a function of the incoming ones
# A is a vector with all incoming messages. At index j_index there is m(j → i)
# ψᵢⱼ are the ones living on the outedges of node i
function f_bp(A::Vector{M2}, wᵢ::Vector{U}, ϕᵢ::Vector{Vector{F}}, 
        ψₙᵢ::Vector{Vector{Matrix{F}}}, j_index::Integer; showprogress=false, 
        svd_trunc::SVDTrunc=TruncThresh(0.0), periodic=false) where {F,U<:BPFactor,M2<:AbstractMPEM2}
    T = length(A[1]) - 1
    @assert all(length(a) == T + 1 for a in A)
    @assert length(wᵢ) == T + 1
    @assert length(ϕᵢ) == T + 1
    q = length(ϕᵢ[1])
    qj = size(ψₙᵢ[j_index][1],1)
    @assert all(length(ϕᵢᵗ) == q for ϕᵢᵗ in ϕᵢ)
    @assert j_index in eachindex(A)
    dt = showprogress ? 1.0 : Inf
    prog = Progress(T - 1, dt=dt, desc="Computing outgoing message")
    notj = eachindex(A)[Not(j_index)]
    xin = Iterators.product((axes(ψₙᵢ[k][1],2) for k in notj)...)
    B = map(1:T+1) do t
        Bᵗ = zeros(reduce(.*, (size(A[k][t])[1:2] for k in notj); init=(1,1))..., q, qj, q)
        @inbounds for xᵢᵗ in 1:q 
            for xₙᵢ₋ⱼᵗ in xin
                @views Aᵗ = kronecker(ones(1,1),
                    (A[k][t][:,:,xₖᵗ,xᵢᵗ] .* ψₙᵢ[k][t][xᵢᵗ, xₖᵗ] for (k, xₖᵗ) in zip(notj,xₙᵢ₋ⱼᵗ))...)
                for xⱼᵗ in 1:qj, xᵢᵗ⁺¹ in 1:q
                    w = ϕᵢ[t][xᵢᵗ]
                    if t <= T || periodic
                        xₙᵢᵗ = [xₙᵢ₋ⱼᵗ[1:j_index-1]..., xⱼᵗ, xₙᵢ₋ⱼᵗ[j_index:end]...]
                        w *= wᵢ[t](xᵢᵗ⁺¹, xₙᵢᵗ, xᵢᵗ)
                    end
                    if !iszero(w)
                        Bᵗ[:, :, xᵢᵗ, xⱼᵗ, xᵢᵗ⁺¹] .+= Aᵗ .* w
                    end
                end
            end
        end
        any(isnan, Bᵗ) && @error "NaN in tensor at time $t"
        next!(prog, showvalues=[(:t, "$t/$T")])
        Bᵗ
    end

    mpem3from2(eltype(A))(B), 0.0
end

# compute outgoing message to dummy neighbor to get the belief
function f_bp_dummy_neighbor(A::Vector{<:AbstractMPEM2}, 
        wᵢ::Vector{U}, ϕᵢ::Vector{Vector{F}}, ψₙᵢ::Vector{Vector{Matrix{F}}};
        showprogress=false, svd_trunc::SVDTrunc=TruncThresh(0.0), periodic=false) where {F,U<:BPFactor}
    q = length(ϕᵢ[1])
    T = length(ϕᵢ) - 1
    @assert all(length(a) == T + 1 for a in A)
    @assert length(wᵢ) == T + 1
    dt = showprogress ? 1.0 : Inf
    prog = Progress(T - 1, dt=dt, desc="Computing outgoing message")
    xin = Iterators.product((axes(ψₙᵢ[k][1],2) for k in eachindex(A))...)
    B = map(1:T+1) do t
        Bᵗ = zeros(reduce(.*, (size(A[k][t])[1:2] for k in eachindex(A)); init=(1,1))..., q, 1, q)
        @inbounds for xᵢᵗ in 1:q
            for xₙᵢᵗ in xin
                @views Aᵗ = kronecker(ones(1,1),
                    (A[k][t][:,:,xₖᵗ,xᵢᵗ] .* ψₙᵢ[k][t][xᵢᵗ, xₖᵗ] for (k, xₖᵗ) in pairs(xₙᵢᵗ))...)
                for xᵢᵗ⁺¹ in 1:q
                    w = ϕᵢ[t][xᵢᵗ]
                    if t <= T || periodic
                        w *= wᵢ[t](xᵢᵗ⁺¹, collect(xₙᵢᵗ), xᵢᵗ)
                    end
                    if !iszero(w)
                        Bᵗ[:, :, xᵢᵗ, 1, xᵢᵗ⁺¹] .+= Aᵗ .* w
                    end
                end
            end
        end
        any(isnan, Bᵗ) && println("NaN in tensor at time $t")
        next!(prog, showvalues=[(:t, "$t/$T")])
        Bᵗ
    end

    mpem3from2(eltype(A))(B), 0.0
end

function pair_belief_as_mpem(Aᵢⱼ::M2, Aⱼᵢ::M2, ψᵢⱼ) where {M2<:AbstractMPEM2}
    A = map(zip(Aᵢⱼ, Aⱼᵢ, ψᵢⱼ)) do (Aᵢⱼᵗ, Aⱼᵢᵗ, ψᵢⱼᵗ)
        @cast Aᵗ[(aᵗ,bᵗ),(aᵗ⁺¹,bᵗ⁺¹),xᵢᵗ,xⱼᵗ] := Aᵢⱼᵗ[aᵗ,aᵗ⁺¹,xᵢᵗ, xⱼᵗ] * 
            Aⱼᵢᵗ[bᵗ,bᵗ⁺¹,xⱼᵗ,xᵢᵗ] * ψᵢⱼᵗ[xᵢᵗ, xⱼᵗ]
    end
    return M2(A)
end

# compute bᵢⱼᵗ(xᵢᵗ,xⱼᵗ) from μᵢⱼ, μⱼᵢ, ψᵢⱼ
# also return normalization zᵢⱼ
function pair_belief(Aᵢⱼ::AbstractMPEM2, Aⱼᵢ::AbstractMPEM2, ψᵢⱼ)
    A = pair_belief_as_mpem(Aᵢⱼ, Aⱼᵢ, ψᵢⱼ)
    l = accumulate_L(A)
    marginals(A; l), normalization(A; l)
end