# Compute magnetization for Ising model on infinite diluted graphs at equilibrium.
# Useful to compare with the dynamics at steady state

struct RandomRegular
    k :: Int    
end

struct ErdosRenyi{T<:Real}
    c :: T      # mean connectivity
end

function equilibrium_magnetization(g::RandomRegular, J::Real; β::Real=1.0, h::Real=0.0,
        maxiter=10^3, tol=1e-16, init=100.0*(sign(h)+rand()))
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
            x = xnew
        end
        error("Fixed point iterations did not converge. err=$err")
    end

    ustar = iterate_fixedpoint(f, init; maxiter, tol)
    return abs( tanh(β*(k*ustar-h)/(k-1)) )
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
        nsamples=10^3, rng=Random.GLOBAL_RNG)
    m = zeros(nsamples)
    us = zeros(nsamples)
    
    P = randn(popsize)
    Pnew = copy(P)

    f(us, Js, β, h) = 1/β*sum(atanh(tanh(β*u)*tanh(β*J)) for (u,J) in zip(us,Js); init=0.0) + h

    function iterate_bp!(P, f, pkm1, pJ, β, ph)
        for idx_out in eachindex(P)
            km1 = rand(pkm1)
            idx_in = rand(eachindex(P), km1)
            Js = rand(rng, pJ, km1); h = rand(ph)
            P[idx_out] = f((@view P[idx_in]), Js, β, h)
        end
        nothing
    end

    iters = 0; err = Inf
    prog = Progress(maxiter, desc="Running population dynamics")
    for _ in 1:maxiter
        iterate_bp!(Pnew, f, pkm1, pJ, β, ph)
        err = abs(mean(P) - mean(Pnew)) 
        err < tol && break
        P, Pnew = Pnew, P
        iters += 1
        next!(prog, showvalues=[(:it, "$iters/$maxiter"), 
            (:ε,"$(round(err, digits=ceil(Int, log10(1/tol))))/$tol")])
    end

    iters == maxiter && @warn "Population dynamics did not converge. Error $err"

    for s in 1:nsamples
        k = rand(pk)
        idx_in = rand(eachindex(P), k)
        Js = rand(rng, pJ, k); h = rand(ph)
        u = f(P[idx_in], Js, β, h)
        m[s] = tanh(β*u)
        us[s] = u
    end

    m_avg = mean(abs.(m))
    m_std = std(abs.(m)) / sqrt(length(m))

    return m_avg ± m_std
end