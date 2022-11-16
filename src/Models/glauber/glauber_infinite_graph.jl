function belief(A::MPEM2)
    b_pair = MatrixProductBP.pair_belief(A, A)[1]
    b_site = map(b_pair) do bᵢⱼᵗ
        bᵢᵗ = vec(sum(bᵢⱼᵗ, dims=2))
    end
end

function onebpiter_infinite_graph(A::MPEM2, k::Integer, pᵢ⁰, wᵢ, ϕᵢ, ψₙᵢ;
        svd_trunc::SVDTrunc=TruncThresh(1e-6))
    
    B = f_bp_glauber(fill(A, k), pᵢ⁰, wᵢ, ϕᵢ, ψₙᵢ, 1)
    C = mpem2(B)
    A_new = sweep_RtoL!(C; svd_trunc)
    normalize_eachmatrix!(A_new)
    A_new, belief(A_new)
end

function iterate_glauber_infinite_graph(T::Integer, k::Integer, pᵢ⁰, wᵢ;
        ϕᵢ = fill(ones(q_glauber), T),
        ψₙᵢ = fill(fill(ones(q_glauber,q_glauber), T), k),
        svd_trunc::SVDTrunc=TruncThresh(1e-6), maxiter=5, tol=1e-5,
        showprogress=true)

    A = mpem2(q_glauber, T)
    Δs = fill(NaN, maxiter)
    b = belief(A)
    dt = showprogress ? 0.1 : Inf
    prog = Progress(maxiter; dt, desc="Iterating BP on infinite graph")
    for it in 1:maxiter
        A, b_new = onebpiter_infinite_graph(A, k, pᵢ⁰, wᵢ, ϕᵢ, ψₙᵢ; svd_trunc)
        Δ = maximum(abs, bb_new[1] - bb[1] for (bb_new, bb) in zip(b_new, b))
        Δs[it] = Δ
        Δ < tol && return A, it, Δs, b
        b, b_new = b_new, b
        rounded_Δ = round(Δ, digits=ceil(Int,log(tol)))
        next!(prog, showvalues=[(:iter, "$it/$maxiter"), (:Δ,"$rounded_Δ/$tol")])
    end
    A, maxiter, Δs, b
end

function glauber_infinite_graph(T::Integer, k::Integer, m⁰::Real;
        β::Real=1.0, J::Real=1.0, h::Real=0.0,
        svd_trunc::SVDTrunc=TruncThresh(1e-6), maxiter=5, tol=1e-5,
        showprogress=true)
    pᵢ⁰ = [(1+m⁰)/2, (1-m⁰)/2]
    wᵢ = fill(HomogeneousGlauberFactor(fill(J, k), h, β), T)
    A, maxiter, Δs, b = iterate_glauber_infinite_graph(T, k, pᵢ⁰, wᵢ; 
        svd_trunc, maxiter, tol, showprogress)
end