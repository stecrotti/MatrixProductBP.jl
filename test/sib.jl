using PyCall, SparseArrays, IndexedGraphs

ENV["PYTHON"] = "/home/crotti/miniconda3/bin/python"

@pyimport sib

# convert ϕ's into observations for sib
function sib_SI_tests(ϕ::Vector{Vector{Vector{F}}}) where F
    tests = []
    for i in eachindex(ϕ)
        for t in eachindex(ϕ[i])
            ϕᵢᵗ = ϕ[i][t]
            # uniform <=> no observation
            if all(isequal(ϕᵢᵗ[1]), ϕᵢᵗ)
                nothing
            else
                push!(tests, (i-1, sib.Test(ϕᵢᵗ..., 0), t+1))
            end
        end
    end
    tests
end


function sib_SI(T::Integer, g::IndexedGraph, ϕ, p⁰, λ;
        maxiter = 400, tol = 1e-14)
    γ = p⁰[1][2]
    @assert all(isequal(γ), pᵢ⁰[2] for pᵢ⁰ in p⁰)

    contacts = [(src(e)-1, dst(e)-1, t, λ) for t in 1:T for e in edges(g)]
    append!(contacts, [(dst(e)-1, src(e)-1, t, λ) for t in 1:T for e in edges(g)])
    sort!(contacts, by=x->x[3])
    tests = sib_SI_tests(ϕ)
    prob_sus = 0.5
    prob_seed = γ
    pseed = prob_seed / (2 - prob_seed)
    psus = prob_sus * (1 - pseed)
    params = sib.Params(prob_r=sib.Exponential(1e-100), pseed=pseed, psus=psus, 
        pautoinf=0.0)
    f = sib.FactorGraph(contacts=contacts, tests=tests, params=params)
    sib.iterate(f, maxit=maxiter, tol=tol)
    sib.iterate(f, maxit=maxiter, damping=0.5, tol=tol)
    sib.iterate(f, maxit=maxiter, damping=0.9, tol=tol)

    marginals_sib = map(f.nodes) do i
        map(1:T ) do t 
            m = sib.marginal_t(i, t)
            [m[1], m[2]]
        end
    end
    return marginals_sib
end