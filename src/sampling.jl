# as in https://doi.org/10.1103/PhysRevLett.114.248701
# draws samples from the prior and weights them with their likelihood
struct SoftMarginSampler{B<:MPBP, F<:AbstractFloat}
    bp :: B
    X  :: Vector{Matrix{Int}}
    w  :: Vector{F}

    function SoftMarginSampler(bp::MPBP{q,T,F,U}, 
            X::Vector{Matrix{Int}}, w::Vector{F}) where{q,T,F,U}

        N = nv(bp.g)
        @assert length(X) == length(w)
        @assert all(≥(0), w)
        @assert all(x -> size(x) == (N, T+1), X)

        new{MPBP{q,T,F,U}, F}(bp, X, w)
    end
end

function SoftMarginSampler(bp::MPBP)
    X = Matrix{Int}[]
    w = zeros(0)
    SoftMarginSampler(bp, X, w)
end

# a sample with its weight
function onesample!(x::Matrix{Int}, bp::MPBP{q,T,F,U};
        rng = GLOBAL_RNG) where {q,T,F,U}
    @unpack g, w, ϕ, ψ, p⁰, μ = bp
    N = nv(bp.g)
    @assert size(x) == (N , T+1)
    logl = 0.0

    for i in 1:N
        xᵢ⁰ = sample_noalloc(rng, p⁰[i])
        x[i, 1] = xᵢ⁰
        logl += log( ϕ[i][1][xᵢ⁰] )
    end

    for t in 1:T
        for i in 1:N
            ∂i = neighbors(bp.g, i)
            p = [w[i][t](xx, x[∂i, t], x[i, t]) for xx in 1:q]
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
function onesample(bp::MPBP{q,T,F,U}; kw...) where {q,T,F,U}  
    N = nv(bp.g)
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
    N = nv(bp.g); T = getT(bp); q = getq(bp)
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
            x = [[xx[i,t], xx[i,u]] for xx in X]
            mtu_avg = zeros(q, q)
            for i in eachindex(x)
                xx = x[i]
                mtu_avg[xx[1], xx[2]] += wv[i] / wv.sum
            end
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
        ϕ[i][t] .= [x==X[i,t] ? softone : softzero for x in eachindex(ϕ[i][t])]
    end
    ϕ
end

# draw 1 sample from the prior, observe something and return the sample
function draw_node_observations!(bp::MPBP, nobs::Integer; kw...)
    X, _ = onesample(bp)
    draw_node_observations!(bp.ϕ, X, nobs; kw...)
    X
end

#### OLD

# # return a sampled trajectory. Better use `SoftMarginSampler`
# function sample!(X, gl::Glauber{T,N,F}) where {T,N,F}
#     @assert is_free_dynamics(gl)
#     @unpack ising = gl

#     # t=0
#     for i in 1:N
#         p = p⁰[i][2]
#         X[1,i] = rand(Bernoulli(p)) + 1
#     end

#     for t in 1:T
#         for i in 1:N
#             ∂i = neighbors(ising.g, i)
#             p = local_w(ising.g, ising.J, ising.h, i, 2, 
#                     X[t,∂i], ising.β)
#             X[t+1,i] = rand(Bernoulli(p)) + 1
#         end
#     end
#     X
# end
# function _sample(gl::Glauber{T,N,F}) where {T,N,F}
#     X = zeros(Int, T+1, N)
#     sample!(X, gl)
# end