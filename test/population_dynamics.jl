@testset "Population dynamics (Glauber infinite regular)" begin
    atol = 1e-3
    rtol = 1e-1

    T = 4
    k = 3
    m⁰ = 0.6
    β = 0.8
    p = 0.2

    popsize = 3
    d = 10
    maxiter = 30
    K = 30
    σ = 1/60
    P = 2.0

    ϕᵢ = [t == 0 ? [(1-m⁰)/2, (1+m⁰)/2] : ones(2) for t in 0:T]
    ψ_neutral = [ones(2,2) for t in 0:T]

    prob_degree = Dirac(k)
    prob_J = Dirac(1.0)
    prob_h = Dirac(0.0)

    wᵢ = fill(DampedFactor(HomogeneousGlauberFactor(rand(prob_J), rand(prob_h), β), p), T+1)
    prob_w(w; d, rng=Random.GLOBAL_RNG) = w

    wᵢ_fourier = fill(FourierGlauberFactor([rand(prob_J) for _ in 1:k], rand(prob_h), β; K, σ, P, p), T+1)
    prob_w_fourier(w; d, rng=Random.GLOBAL_RNG) = w

    μ_pop = map(1:popsize) do p
        μ = rand_mpem2(2, 2, T)
        normalize!(μ)
        μ
    end |> AtomicVector
    bs = Vector{Vector{Float64}}[] |> AtomicVector
    bs2times =  Matrix{Matrix{Float64}}[] |> AtomicVector

    μ_pop_fourier = map(1:popsize) do p
        μ = rand_mpem2(ComplexF64, 2, 2, T)
        normalize!(μ)
        μ
    end |> AtomicVector
    bs_fourier = Vector{Vector{Float64}}[] |> AtomicVector
    bs2times_fourier =  Matrix{Matrix{Float64}}[] |> AtomicVector

    function stats!(statvecs, wᵢ, μ, μin, b, f)
        @assert length(statvecs) == 2
        (bs, bs2times) = statvecs
        belief = [real.(m) for m in marginals(b)]
        belief2times = [real.(m) for m in twovar_marginals(b)]
        push!(bs, belief)
        push!(bs2times, belief2times)
    end

    rng = MersenneTwister(123)

    iterate_popdyn!(μ_pop, wᵢ, prob_degree, prob_w, (bs, bs2times); ϕ=ϕᵢ, maxiter, svd_trunc=TruncBond(d), stats=stats!, rng)
    iterate_popdyn!(μ_pop_fourier, wᵢ_fourier, prob_degree, prob_w_fourier, (bs_fourier, bs2times_fourier); ϕ=ϕᵢ, maxiter, svd_trunc=TruncBond(d), stats=stats!, rng)

    Nmc = 10^3
    g = random_regular_graph(Nmc, k) |> IndexedBiDiGraph
    J = 1.0
    h = 0.0
    w_mc = [fill(DampedFactor(HomogeneousGlauberFactor(J*β, h*β), p), T+1) for i in vertices(g)]
    bp_mc = mpbp(Float64, g, w_mc, fill(2, nv(g)), T; ϕ = fill(ϕᵢ, Nmc))
    sms = SoftMarginSampler(bp_mc)

    X = zeros(Int, Nmc, T+1)
    autocorrs_mc = [zeros(T+1) for _ in 1:Nmc]
    means_mc = [zeros(T+1) for _ in 1:Nmc]
    energy_mc = zeros(T)

    nsamples = 10^4
    for samp in 1:nsamples
        onesample!(X, bp_mc)
        for i in 1:Nmc
            autocorrs_mc[i] .+= potts2spin.(X[i,:]) .* potts2spin(X[i,end])
            means_mc[i] .+= potts2spin.(X[i,:])
        end
    end

    autocorrs_mc ./= nsamples
    means_mc ./= nsamples
    autocorrs_mc .-= means_mc .* [x[end] for x in means_mc]
    autocorr_mc = mean([abs.(x) for x in autocorrs_mc])
    m_mc = mean(means_mc)

    ns = 20
    range = length(bs)+1-min(ns, length(bs)):length(bs)
    ms = [expectation.(potts2spin, b) for b in bs[range]]
    m = mean(ms)
    σ = std(ms) ./ sqrt(length(ms))
    range_fourier = length(bs_fourier)+1-min(ns, length(bs_fourier)):length(bs_fourier)
    ms_fourier = [expectation.(potts2spin, b) for b in bs_fourier[range_fourier]]
    m_fourier = mean(ms_fourier)
    σ_fourier = std(ms_fourier) ./ sqrt(length(ms_fourier))

    rs = [expectation.(potts2spin, btu) for btu in bs2times[range]]
    cs = MatrixProductBP.covariance.(rs, ms)
    c_avg = mean(abs.(x) for x in cs)
    c_std = std(cs) ./ sqrt(length(cs))
    rs_fourier = [expectation.(potts2spin, btu) for btu in bs2times_fourier[range_fourier]]
    cs_fourier = MatrixProductBP.covariance.(rs_fourier, ms_fourier)
    c_avg_fourier = mean(abs.(x) for x in cs_fourier)
    c_std_fourier = std(cs_fourier) ./ sqrt(length(cs_fourier))

    @testset "Observables" begin
        @test isapprox(m_mc, m; atol, rtol)
        @test isapprox(m_mc, m_fourier; atol, rtol)
        @test isapprox(autocorr_mc[1:end-1], c_avg[1:end-1,end]; atol, rtol)
        @test isapprox(autocorr_mc[1:end-1], c_avg_fourier[1:end-1,end]; atol, rtol)
    end
end