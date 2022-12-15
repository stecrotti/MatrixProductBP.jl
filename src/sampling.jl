# as in https://doi.org/10.1103/PhysRevLett.114.248701
# draw samples from the prior and weight them with their likelihood
struct SoftMarginSampler{B<:MPBP, F<:AbstractFloat}
    bp :: B
    X  :: Vector{Matrix{Int}}
    w  :: Vector{F}

    function SoftMarginSampler(bp::TBP, 
            X::Vector{Matrix{Int}}, w::Vector{F}) where{TBP<:MPBP, F}

        N = nv(bp.g); T = getT(bp)
        @assert length(X) == length(w)
        @assert all(≥(0), w)
        @assert all(x -> size(x) == (N, T+1), X)

        new{TBP, F}(bp, X, w)
    end
end

function SoftMarginSampler(bp::MPBP)
    X = Matrix{Int}[]
    w = zeros(0)
    SoftMarginSampler(bp, X, w)
end

# a sample with its weight
function onesample!(x::Matrix{Int}, bp::MPBP{G,F,U};
        rng = GLOBAL_RNG) where {G,F,U}
    @unpack g, w, ϕ, ψ, μ = bp
    N = nv(bp.g); T = getT(bp); q = nstates(U)
    @assert size(x) == (N , T+1)
    logl = 0.0

    for i in 1:N
        @assert sum(ϕ[i][1]) == 1
        xᵢ⁰ = sample_noalloc(rng, ϕ[i][1])
        x[i, 1] = xᵢ⁰
        logl += log( ϕ[i][1][xᵢ⁰] )
    end

    for t in 1:T
        for i in 1:N
            ∂i = neighbors(bp.g, i)
            @views p = [w[i][t](xx, x[∂i, t], x[i, t]) for xx in 1:q]
            xᵢᵗ = sample_noalloc(rng, p)
            x[i, t+1] = xᵢᵗ
            logl += log( ϕ[i][t+1][xᵢᵗ] )
        end
    end
    for t in 1:T+1
        for (i, j, ij) in edges(bp.g)
            logl += 1/2 * log( ψ[ij][t][x[i,t], x[j,t]] )
        end
    end
    return x, exp(logl)
end
function onesample(bp::MPBP; kw...)
    N = nv(bp.g); T = getT(bp)
    x = zeros(Int, N, T+1)
    onesample!(x, bp; kw...)
end

function sample!(sms::SoftMarginSampler, nsamples::Integer;
        showprogress::Bool=true)

    dt = showprogress ? 0.1 : Inf
    prog = Progress(nsamples, desc="SoftMargin sampling"; dt)
    T = getT(sms.bp); N = getN(sms.bp)
    X = [zeros(Int, N, T+1) for _ in 1:nsamples]
    w = zeros(nsamples)
    for n in 1:nsamples
        _, w[n] = onesample!(X[n], sms.bp)
        next!(prog)
    end 
    append!(sms.X, X)
    append!(sms.w, w)
    
    sms
end

function sample(bp::MPBP, nsamples::Integer; showprogress::Bool=true)
    sms = SoftMarginSampler(bp)
    sample!(sms, nsamples; showprogress)
end

# return a (T+1) by N matrix, with uncertainty estimates
function marginals(sms::SoftMarginSampler) 
    @unpack bp, X, w = sms
    N = nv(bp.g); T = getT(bp); q = nstates(bp)
    marg = [[zeros(Measurement, q) for t in 0:T] for i in 1:N]
    @assert all(>=(0), w)
    wv = weights(w)
    nsamples = length(X)

    for i in 1:N
        for t in 1:T+1
            x = [xx[i, t] for xx in X]
            mit_avg = proportions(x, q, wv)
            # avoid numerical errors yielding probabilities > 1
            mit_avg = map(x -> x≥1 ? 1 : x, mit_avg)
            mit_var = mit_avg .* (1 .- mit_avg) ./ nsamples
            marg[i][t] .= mit_avg .± sqrt.( mit_var )
        end
    end

   return marg
end

function autocorrelations(sms::SoftMarginSampler; showprogress::Bool=true)
    @unpack bp, X, w = sms
    N = nv(bp.g); T = getT(bp); q = getq(bp)
    r = [fill(zero(Measurement), T+1, T+1) for i in 1:N]
    @assert all(>=(0), w)
    wv = weights(w)
    nsamples = length(X)
    U = getU(sms.bp)
    dt = showprogress ? 0.1 : Inf
    prog = Progress(N, desc="Autocorrelations from Soft Margin"; dt)

    for i in 1:N
        for u in axes(r[i], 2), t in 1:u-1
            mtu_avg = zeros(q, q)
            for (n, x) in enumerate(X)
                mtu_avg[x[i,t], x[i,u]] += wv[n]
            end
            mtu_avg ./= wv.sum
            mtu_var = mtu_avg .* (1 .- mtu_avg) ./ nsamples
            r[i][t,u] = marginal_to_expectation(mtu_avg .± sqrt.( mtu_var ), U)  
        end
        next!(prog)
    end
    r
end

function autocovariances(sms::SoftMarginSampler; showprogress::Bool=true,
        r = autocorrelations(sms; showprogress), m = marginals(sms))
    U = getU(sms.bp)
    μ = [marginal_to_expectation.(mᵢ, U) for mᵢ in m] 
    _autocovariances(r, μ)
end


# draw `nobs` observations from the prior
# flag `last_time` draws all observations from time T
function draw_node_observations!(ϕ::Vector{Vector{Vector{F}}}, 
        X::Matrix{<:Integer}, nobs::Integer; softinf::Real=Inf,
        last_time::Bool=false, rng=GLOBAL_RNG) where {F<:Real}
    N, T = size(X) .- (0, 1)
    it = if last_time
        sample(rng,  collect(Iterators.product(T+1:T+1, 1:N)), nobs, replace=false)
    else
        sample(rng,  collect(Iterators.product(1:T+1, 1:N)), nobs, replace=false)
    end
    softone = logistic(log(softinf)); softzero = logistic(-log(softinf))
    for (t, i) in it
        ϕ[i][t] .*= [x==X[i,t] ? softone : softzero for x in eachindex(ϕ[i][t])]
        if t == 1
            ϕ[i][t] ./= sum(ϕ[i][t])
        end
    end
    ϕ
end

# draw 1 sample from the prior, observe something and return the sample
function draw_node_observations!(bp::MPBP, nobs::Integer; kw...)
    X, _ = onesample(bp)
    draw_node_observations!(bp.ϕ, X, nobs; kw...)
    X
end