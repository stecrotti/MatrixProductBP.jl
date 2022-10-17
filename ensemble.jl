import ProgressMeter: Progress, next!
import Statistics: mean

include("mpdbp.jl")

function onestep_rs_deg3(A::MPEM2, pᵢ⁰, ϕᵢ; ε=1e-6,
        mag_old = magnetization_rs_deg3(A, pᵢ⁰, ϕᵢ; ε))

    B = f_bp([A, A, A], pᵢ⁰, wᵢ, ϕᵢ, 1)
    C = mpem2(B)
    GC.gc()
    A_new = sweep_RtoL!(C; ε)
    normalize_eachmatrix!(A_new)
    mag_new = magnetization_rs_deg3(A_new, pᵢ⁰, ϕᵢ; ε)
    Δ = mean(abs, mag_new .- mag_old)
    return A_new, Δ, mag_new
end

function iterate_rs_deg3(A::MPEM2, pᵢ⁰, ϕᵢ; ε=1e-6, maxiter=5, tol=1e-10)
    mag = magnetization_rs_deg3(A, pᵢ⁰, ϕᵢ; ε)

    Δs = zeros(maxiter)
    prog = Progress(maxiter)
    for it in 1:maxiter
        A, Δ, mag = onestep_rs_deg3(A, pᵢ⁰, ϕᵢ, mag_old=mag; ε)
        Δs[it] = Δ
        Δ < tol && return it, mag, Δs
        bd = bond_dims(A)
        next!(prog, showvalues=[(:Δ, Δ), (:bonds, bd)])
        GC.gc()
    end
    return maxiter, mag, Δs
end

function belief_rs_deg3(A::MPEM2, pᵢ⁰, ϕᵢ; ε=1e-6)
    B = f_bp([A, A, A], pᵢ⁰, wᵢ, ϕᵢ)
    C = mpem2(B)
    GC.gc()
    A_new = sweep_RtoL!(C; ε)
    normalize_eachmatrix!(A_new)
    belief = firstvar_marginals(A_new)
    return belief
end

function magnetization_rs_deg3(A::MPEM2, pᵢ⁰, ϕᵢ; ε=1e-6)
    b = belief_rs_deg3(A, pᵢ⁰, ϕᵢ; ε)
    mag = reduce.(-, b)
    return mag
end

