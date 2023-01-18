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
    return tanh(β*(k*ustar-h)/(k-1))
end


function equilibrium_magnetization(g::ErdosRenyi, J::Real; β::Real=1.0, h::Real=0.0,
        popsize=10^3, maxiter=10^3, tol=sqrt(popsize), nsamples=10^3)
    c = g.c
    m = zeros(nsamples)
    
    P = randn(popsize)
    Pnew = copy(P)

    f(us, J, β, h) = 1/β*sum(atanh(tanh(β*u)*tanh(β*J)) for u in us; init=0.0) + h

    function iterate_ER!(P, f, c, J, β, h)
        for idx_out in eachindex(P)
            km1 = rand(Poisson(c))
            idx_in = rand(eachindex(P), km1)
            P[idx_out] = f((@view P[idx_in]), J, β, h)
        end
        nothing
    end

    iters = 0; err = Inf
    for _ in 1:maxiter
        iterate_ER!(Pnew, f, c, J, β, h)
        err = abs(mean(P) - mean(Pnew)) 
        err < tol && break
        P, Pnew = Pnew, P
        iters += 1
    end

    iters == maxiter && @warn "Population dynamics did not converge. Error $err"

    for s in 1:nsamples
        k = rand(Poisson(c))
        idx_in = rand(eachindex(P), k)
        u = f(P[idx_in], J, β, h)
        m[s] = tanh(β*u)
    end

    m_avg = mean(abs.(m))
    m_std = std(abs.(m)) / sqrt(length(m))

    return m_avg ± m_std
end