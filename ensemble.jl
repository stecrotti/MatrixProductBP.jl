import ProgressMeter: Progress, next!
import Statistics: mean

include("mpdbp.jl")

function onestep_rs_deg3(A::MPEM2, pᵢ⁰, wᵢ, ϕᵢ; 
        svd_trunc::SVDTrunc=TruncThresh(1e-6),
        mag_old = message_magnetization_deg3(A),
        showprogress = true)
    B = f_bp([A, A, A], pᵢ⁰, wᵢ, ϕᵢ, 1; showprogress)
    C = mpem2(B; showprogress)
    A_new = sweep_RtoL!(C; svd_trunc)
    normalize_eachmatrix!(A_new)
    mag_new = message_magnetization_deg3(A_new)
    Δ = maximum(abs, mag_new .- mag_old)
    return A_new, Δ, mag_new
end

function iterate_rs_deg3(A::MPEM2, pᵢ⁰, wᵢ, ϕᵢ; 
        svd_trunc::SVDTrunc=TruncThresh(1e-6), maxiter=5, tol=1e-10,
        Δs = zeros(0), verbose=true, showprogress=verbose)
    mag_msg = message_magnetization_deg3(A)

    for it in 1:maxiter
        A, Δ, mag_msg = onestep_rs_deg3(A, pᵢ⁰, wᵢ, ϕᵢ, mag_old=mag_msg; 
            svd_trunc, showprogress)
        push!(Δs, Δ)
        bd = bond_dims(A)
        if verbose
            println()
            println("### iter $it of $maxiter")
            println("Δ: $Δ / $tol")
            println("bonds:\t", bd)
            println()
        end
        Δ < tol && return A, it, Δs
    end
    return A, maxiter, Δs
end

function belief_rs_deg3(A::MPEM2, pᵢ⁰, wᵢ, ϕᵢ; 
    svd_trunc::SVDTrunc=TruncThresh(1e-6), showprogress = true)
    B = f_bp([A, A, A], pᵢ⁰, wᵢ, ϕᵢ; showprogress)
    C = mpem2(B; showprogress)
    A_new = sweep_RtoL!(C; svd_trunc)
    normalize_eachmatrix!(A_new)
    b = firstvar_marginals(A_new)
    return b
end

function message_magnetization_deg3(A::MPEM2)
    m = firstvar_marginals(A)
    mag = reduce.(-, m)
    return mag
end

function magnetization_rs_deg3(A::MPEM2, pᵢ⁰, wᵢ, ϕᵢ; 
    svd_trunc::SVDTrunc=TruncThresh(1e-6), showprogress = true)
    b = belief_rs_deg3(A, pᵢ⁰, wᵢ, ϕᵢ; svd_trunc, showprogress)
    mag = reduce.(-, b)
    return mag
end

