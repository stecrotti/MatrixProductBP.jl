import ProgressMeter: @showprogress
import CavityTools: ExponentialQueue

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
function onesample!(x::Matrix{Int}, bp::MPBP{G,F};
        p⁰ = [ϕᵢ[1] ./ sum(ϕᵢ[1]) for ϕᵢ in bp.ϕ],
        rng = GLOBAL_RNG) where {G,F}
    @unpack g, w, ϕ, ψ, μ = bp
    N = nv(bp.g); T = getT(bp);
    @assert size(x) == (N , T+1)
    logl = 0.0

    for i in 1:N
        xᵢ⁰ = sample_noalloc(rng, p⁰[i])
        x[i, 1] = xᵢ⁰
    end

    for t in 1:T
        for i in 1:N
            ∂i = neighbors(bp.g, i)
            q = nstates(bp, i)
            p = @views (w[i][t](xx, x[∂i, t], x[i, t]) for xx in 1:q)
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
        showprogress::Bool=true, rng = GLOBAL_RNG)

    dt = showprogress ? 0.1 : Inf
    prog = Progress(nsamples, desc="SoftMargin sampling"; dt)
    T = getT(sms.bp); N = getN(sms.bp)
    X = [zeros(Int, N, T+1) for _ in 1:nsamples]
    p⁰ = [ϕᵢ[1] ./ sum(ϕᵢ[1]) for ϕᵢ in sms.bp.ϕ]
    w = zeros(nsamples)
    for n in 1:nsamples
        _, w[n] = onesample!(X[n], sms.bp; p⁰, rng)
        next!(prog)
    end 
    append!(sms.X, X)
    append!(sms.w, w)
    
    sms
end

function sample(bp::MPBP, nsamples::Integer; kw...)
    sms = SoftMarginSampler(bp)
    sample!(sms, nsamples; kw...)
end

# return a (T+1) by N matrix, with uncertainty estimates
function marginals(sms::SoftMarginSampler; showprogress::Bool=true) 
    @unpack bp, X, w = sms
    N = nv(bp.g); T = getT(bp);
    marg = [[zeros(Measurement, nstates(bp,i)) for t in 0:T] for i in 1:N]
    @assert all(>=(0), w)
    wv = weights(w)
    nsamples = length(X)
    prog = Progress(N, desc="Marginals from Soft Margin"; dt=showprogress ? 0.1 : Inf)

    for i in 1:N
        for t in 1:T+1
            x = [xx[i, t] for xx in X]
            mit_avg = proportions(x, nstates(bp,i), wv)
            # avoid numerical errors yielding probabilities > 1
            mit_avg = map(x -> x≥1 ? 1 : x, mit_avg)
            mit_var = mit_avg .* (1 .- mit_avg) ./ nsamples
            marg[i][t] .= mit_avg .± sqrt.( mit_var )
        end
        next!(prog)
    end

   return marg
end

function means(f, sms::SoftMarginSampler)
    b_mc = marginals(sms)
    return [[expectation(f, bb) for bb in b] for b in b_mc]
end

# return a (T+1) by |E| matrix, with uncertainty estimates
function pair_marginals(sms::SoftMarginSampler; showprogress::Bool=true) 
    @unpack bp, X, w = sms
    g = bp.g
    T = getT(bp); E = ne(g)
    marg = [[zeros(Measurement, nstates(bp,i), nstates(bp,j)) for t in 0:T] for (i,j,id) in edges(g)]
    @assert all(>=(0), w)
    wv = weights(w)
    nsamples = length(X)
    prog = Progress(E, desc="Marginals from Soft Margin"; dt=showprogress ? 0.1 : Inf)
    x = zeros(Int, length(X))

    for (i,j,id) in edges(g)
        linear = LinearIndices((1:nstates(bp,i), 1:nstates(bp,j)))
        for t in 1:T+1
            x .= [linear[xx[i, t],xx[j,t]] for xx in X]
            mijt_avg_linear = proportions(x, nstates(bp,i)*nstates(bp,j), wv)
            mijt_avg = reshape(mijt_avg_linear, linear.indices...)
            # avoid numerical errors yielding probabilities > 1
            mijt_avg = map(x -> x≥1 ? 1 : x, mijt_avg)
            mijt_var = mijt_avg .* (1 .- mijt_avg) ./ nsamples
            marg[id][t] .= mijt_avg .± sqrt.( mijt_var )
        end
        next!(prog)
    end

   return marg
end

function autocorrelations(f, sms::SoftMarginSampler; showprogress::Bool=true)
    @unpack bp, X, w = sms
    N = nv(bp.g); T = getT(bp)
    r = [fill(zero(Measurement), T+1, T+1) for _ in 1:N]
    @assert all(>=(0), w)
    wv = weights(w)
    nsamples = length(X)
    dt = showprogress ? 0.1 : Inf
    prog = Progress(N, desc="Autocorrelations from Soft Margin"; dt)

    for i in 1:N
        for u in axes(r[i], 2), t in 1:u-1
            q = nstates(bp, i)
            mtu_avg = zeros(q, q)
            for (n, x) in enumerate(X)
                mtu_avg[x[i,t], x[i,u]] += wv[n]
            end
            mtu_avg ./= wv.sum
            mtu_var = mtu_avg .* (1 .- mtu_avg) ./ nsamples
            r[i][t,u] = expectation(x->f(x, 0), mtu_avg .± sqrt.( mtu_var ))  
        end
        next!(prog)
    end
    r
end

function autocovariances(f, sms::SoftMarginSampler; showprogress::Bool=true,
        r = autocorrelations(f, sms; showprogress), m = marginals(sms))
    μ = [expectation.(x->f(x, i), m[i]) for i in eachindex(m)] 
    covariance.(r, μ)
end


# draw `nobs` observations from the prior
# flag `last_time` draws all observations from time T
# return also the observed (site,time) pairs
function draw_node_observations!(ϕ::Vector{Vector{Vector{F}}}, 
        X::Matrix{<:Integer}, nobs::Integer; softinf::Real=Inf, last_time::Bool=false,
        times=((last_time ? size(X,2) : 1):size(X,2)), rng=GLOBAL_RNG) where {F<:Real}
    N = size(X,1)
    observed = sample(rng,  collect(Iterators.product(1:N, times)), nobs, replace=false)
    sort!(observed)
    softone = logistic(log(softinf)); softzero = logistic(-log(softinf))
    for (i,t) in observed
        ϕ[i][t] .*= [x==X[i,t] ? softone : softzero for x in eachindex(ϕ[i][t])]
    end
    ϕ, observed
end

# draw 1 sample from the prior, observe something and return the sample
function draw_node_observations!(bp::MPBP, nobs::Integer; rng=GLOBAL_RNG, kw...)
    X, _ = onesample(bp; rng)
    _, observed = draw_node_observations!(bp.ϕ, X, nobs; rng, kw...)
    X, observed
end


# A continous-time sampler for SIS
# g = contact network
# λ = rate of infection
# μ = rate of recovery
# T = final time
function simulate_queue_sis!(x, g, P0, λ, μ, T;
    stats = (t, i, x) -> println("$t $i $(x[i])"),
    Q = ExponentialQueue(length(x)))
    t = 0.0
    @assert eachindex(x) == vertices(g)
    fill!(x, false)
    empty!(Q)
    for (i,p) in pairs(P0)
        if rand() < p
            Q[i] = Inf
        end
    end
    while !isempty(Q)
        i, Δt = pop!(Q)
        t += Δt
        t > T && break
        x[i] ⊻= true
        stats(t, i, x)
        if x[i] == 1
            for j in neighbors(g, i)
                if x[j] == 0
                    Q[j] = haskey(Q, j) ? Q[j] + λ : λ
                end
            end
            Q[i] = μ
        else
            s = 0.0
            for j in neighbors(g, i)
                if x[j] == 0
                    Q[j] -= λ
                else
                    s += λ
                end
            end
            Q[i] = s
        end
    end
    x
end


function continuous_sis_sampler(sis, T, λ, ρ; nsamples = 10^5, sites=1:nv(sis.g), Δt=T/200,
        discard_dead_epidemics=false)
    K = floor(Int, T/Δt)+1
    N = nv(sis.g)
    av = [zeros(K) for _ in 1:N]
    va = [zeros(K) for _ in 1:N]
    ni = [zeros(Int, K) for _ in 1:N]
    function stats(t, i, x)
        if i ∈ sites
            k = ceil(Int, t/Δt) + 1
            ni[i][k] += 2x[i]-1
        end
    end

    P0 = [p[1][2] for p in sis.ϕ]
    x = falses(N);
    Q = ExponentialQueue(N)
    ndiscarded = 0
    @showprogress for _ = 1:nsamples
        for nik in ni; fill!(nik, 0); end
        simulate_queue_sis!(x, sis.g, P0, λ, ρ, T; stats, Q)
        if discard_dead_epidemics && all(isequal(false), x)
            ndiscarded += 1
        else
            for i in sites
                s = 0
                for (k,v) in pairs(ni[i])
                    s += v
                    av[i][k] += s
                    va[i][k] += s^2
                end
            end
        end
    end
    for i in 1:N
        av[i] ./= (nsamples - ndiscarded)
        va[i] ./= (nsamples - ndiscarded)
        va[i] .-= av[i] .^ 2
        va[i] .= sqrt.(va[i])
    end
    (;mean=av, std=va)
end