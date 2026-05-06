"generate a factor (possibly dependent on random parameters)"
prob_w(w::Vector{<:BPFactor}; kw...) = error("Not implemented")

function default_stats!((bs,), wᵢ, μ, μin, b, f)
    push!(bs, [real.(m) for m in marginals(b)])
end

function onebpiter_popdyn!(μ, wᵢ, dᵢ, ns, ϕᵢ, ψ, T, svd_trunc)
    μin = deepcopy(μ)
    C, full, _ = compute_prob_ys(wᵢ, ns, μ, ψ, T, svd_trunc)
    sumlogzᵢ₂ⱼ = 0.0
    for j in 1:dᵢ
        B = f_bp_partial_ij(C[j], wᵢ, ϕᵢ, dᵢ-1, ns, j)
        μ[j] = orthogonalize_right!(mpem2(B); svd_trunc)
        sumlogzᵢ₂ⱼ += normalize!(μ[j])
    end
    B = f_bp_partial_i(full, wᵢ, ϕᵢ, dᵢ)
    b = B |> mpem2 |> marginalize
    logzᵢ = normalize!(b)

    f = dᵢ == 0 ? 0.0 : real((dᵢ/2-1)*logzᵢ - (1/2)*sumlogzᵢ₂ⱼ)
    return μ, μin, b, f
end

"""
    iterate_popdyn!(μ_pop::AbstractVector{<:TensorTrain{F,N}}, w::Vector{<:RecursiveBPFactor}, prob_degree, prob_w, statvecs=(); kw...) where {F<:Number, N}

Performs population dynamics over a population of messages `μ_pop`. The function iteratively updates the messages based on randomly sampled factors and degrees, and accumulates statistics in `statvecs`.
"""
function iterate_popdyn!(μ_pop::AbstractVector{<:TensorTrain{F,N}}, w::Vector{<:RecursiveBPFactor}, prob_degree, prob_w, statvecs=(bs,);
    maxiter=10^2, svd_trunc=TruncThresh(1e-6), ns::Integer=2, T::Integer=length(μ_pop[1])-1,
    ϕ=[ones(2) for t in 0:T], ψ=[ones(2,2) for t in 0:T], stats=default_stats!) where {F<:Number, N}
    @showprogress Threads.@threads for n in 1:maxiter
        dᵢ = rand(prob_degree)
        wᵢ = prob_w(w; d=dᵢ)
        indices = rand(eachindex(μ_pop), dᵢ)
        μ, μin, b, f = onebpiter_popdyn!(μ_pop[indices], wᵢ, dᵢ, ns, ϕ, fill(ψ, dᵢ), T, svd_trunc)
        
        stats(statvecs, wᵢ, μ, μin, b, f)
        μ_pop[indices] = μ
    end
end