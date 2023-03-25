# Compute magnetization for Ising model on infinite diluted graphs at equilibrium.
# Useful to compare with the dynamics at steady state

struct RandomRegular
    k :: Int    
end

struct ErdosRenyi{T<:Real}
    c :: T      # mean connectivity
end

function equilibrium_magnetization(g::RandomRegular, J::Real; β::Real=1.0, h::Real=0.0,
        maxiter=10^3, tol=1e-16, init=100.0*(sign(h)+rand()), damp=0.0)
    k = g.k
    f(u) = (k-1)/β *atanh(tanh(β*u)*tanh(β*J)) + h

    function iterate_fixedpoint(f, init; maxiter=10^3, tol=1e-16)
        x = init
        err = Inf
        for _ in 1:maxiter
            xnew = f(x)
            abs(xnew) < tol && return 0.0
            err = abs((x-xnew)/x) 
            err < tol && return x
            x = (1-damp)*xnew + damp*x
        end
        error("Fixed point iterations did not converge. err=$err")
    end

    ustar = iterate_fixedpoint(f, init; maxiter, tol)
    return abs( tanh(β*(k*ustar-h)/(k-1)) )
end

# A callback to print info and save stuff during the iterations 
struct CB_Pop{TP<:ProgressUnknown, F}
    prog :: TP
    f    :: F
    m    :: Vector{Float64} 
    Δs   :: Vector{Float64}

    function CB_Pop(; showprogress::Bool=true, f::F=mean, info="") where F
        dt = showprogress ? 0.1 : Inf
        isempty(info) || (info *= "\n")
        prog = ProgressUnknown(desc=info*"Running PopDyn: iter", dt=dt)
        TP = typeof(prog)

        m = [Inf]
        Δs = zeros(0)
        new{TP,F}(prog, f, m, Δs)
    end
end

function (cb::CB_Pop)(P::Vector{Float64}, it::Integer, maxiter::Integer, tol)
    mnew = cb.f(P)
    Δ = abs(mnew - cb.m[end]) 
    push!(cb.Δs, Δ)
    push!(cb.m, mnew)
    next!(cb.prog, showvalues=[(:it, "$it/$maxiter"), 
        (:ε,"$(round(Δ, digits=ceil(Int, log10(1/tol))))/$tol")])
    return Δ
end


function equilibrium_magnetization(g::ErdosRenyi; kw...)
    equilibrium_magnetization(Poisson(g.c), Poisson(g.c); kw...)
end

function equilibrium_magnetization(g::RandomRegular; kw...)
    equilibrium_magnetization(Dirac(g.k-1), Dirac(g.k); kw...)
end

# pkm1 is the residual probability: prob that a random edge is incident on a var of degree k
# pk is the degree profile
function equilibrium_magnetization(pkm1::Distribution, pk::Distribution; 
        pJ::Distribution=Dirac(1.0), β::Real=1.0,
        ph::Distribution=Dirac(0.0), popsize=10^3, maxiter=10^3, tol=0.1/sqrt(popsize), 
        nsamples=10^3, rng=GLOBAL_RNG,
        P = randn(rng, popsize), cb=CB_Pop())
    
    m = zeros(nsamples)
    Pnew = copy(P)

    f(us, Js, β, h) = 1/β*sum(atanh(tanh(β*u)*tanh(β*J)) for (u,J) in zip(us,Js); init=0.0) + h

    function iterate_bp!(P, f, pkm1, pJ, β, ph, indices)
        for idx_out in indices
            km1 = rand(rng, pkm1)
            idx_in = rand(rng, eachindex(P), km1)
            Js = rand(rng, pJ, km1); h = rand(rng, ph)
            P[idx_out] = f((@view P[idx_in]), Js, β, h)
        end
        nothing
    end

    Δ = Inf
    indices = collect(eachindex(P))
    for it in 1:maxiter
        iterate_bp!(Pnew, f, pkm1, pJ, β, ph, indices)
        Δ = cb(Pnew, it, maxiter, tol)
        Δ < tol && break
        P, Pnew = Pnew, P
        shuffle!(rng, indices)
    end

    Δ < tol || @warn "Population dynamics did not converge. Error $Δ"

    for s in 1:nsamples
        k = rand(rng, pk)
        idx_in = rand(rng, eachindex(P), k)
        Js = rand(rng, pJ, k); h = rand(ph)
        u = f(P[idx_in], Js, β, h)
        m[s] = tanh(β*u)
    end

    m_avg = abs(mean(m))
    m_std = std(m) / sqrt(length(m))

    return m_avg ± m_std
end
