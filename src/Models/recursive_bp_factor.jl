""""
For a `w::U` where `U<:RecursiveBPFactor`, outgoing messages can be computed recursively
A `<:RecursiveBPFactor` must implement: `nstates`, `prob_y`, `prob_xy` and `prob_yy`
Optionally, it can also implement `prob_y_partial` and `(w::U)(xᵢᵗ⁺¹, xₙᵢᵗ, xᵢᵗ)`
"""
abstract type RecursiveBPFactor <: BPFactor; end

#### the next five methods are the minimal needed interface for a new <:RecursiveBPFactor

"Number of states for aux variable which accumulates the first `l` neighbors"
nstates(::Type{<:RecursiveBPFactor}, l::Integer) = error("Not implemented")

"P(xᵢᵗ⁺¹|xᵢᵗ, xₖᵗ, yₙᵢᵗ, dᵢ)
Might depend on the degree `dᵢ` because of the (possible) change of variable from 
    y ∈ {1,2,...} to its physical value, e.g. {-dᵢ,...,dᵢ}"
prob_y(wᵢ::U, xᵢᵗ⁺¹, xᵢᵗ, yₙᵢᵗ, dᵢ) where {U<:RecursiveBPFactor} = error("Not implemented")

"P(yₖᵗ| xₖᵗ, xᵢᵗ)"
prob_xy(wᵢ::RecursiveBPFactor, yₖ, xₖ, xᵢ) = error("Not implemented")
prob_xy(wᵢ::RecursiveBPFactor, yₖ, xₖ, xᵢ, k) = prob_xy(wᵢ, yₖ, xₖ, xᵢ)

"P(yₐᵦ|yₐ,yᵦ,xᵢᵗ)"
prob_yy(wᵢ::RecursiveBPFactor, y, y1, y2, xᵢ, d1, d2) = prob_yy(wᵢ::RecursiveBPFactor, y, y1, y2, xᵢ)
prob_yy(wᵢ::RecursiveBPFactor, y, y1, y2, xᵢ) = error("Not implemented")


##############################################
#### the next methods are optional


"P(xᵢᵗ⁺¹|xᵢᵗ, xₙᵢᵗ, d)"
function (wᵢ::RecursiveBPFactor)(xᵢᵗ⁺¹::Integer, xₙᵢᵗ::AbstractVector{<:Integer}, 
    xᵢᵗ::Integer)
    U = typeof(wᵢ)
    d = length(xₙᵢᵗ)
    Pyy = fill(1.0, 1)
    for k in 1:d
        Pyy = [sum(prob_yy(wᵢ, y, y1, y2, xᵢᵗ, 1, k-1)*prob_xy(wᵢ, y1, xₙᵢᵗ[k], xᵢᵗ,k)*Pyy[y2]
                   for y1 in 1:nstates(U,1), y2 in 1:nstates(U,k-1)) 
               for y in 1:nstates(U,k)]
    end
    sum(Pyy[y] * prob_y(wᵢ, xᵢᵗ⁺¹, xᵢᵗ, y, d) for y in eachindex(Pyy))
end

function prob_y_partial(wᵢ::U, xᵢᵗ⁺¹, xᵢᵗ, xₖᵗ, y1, d, k) where {U<:RecursiveBPFactor}
    sum(prob_y(wᵢ, xᵢᵗ⁺¹, xᵢᵗ, yᵗ, d + 1) * 
        prob_xy(wᵢ, y2, xₖᵗ, xᵢᵗ,k) * 
        prob_yy(wᵢ, yᵗ, y1, y2, xᵢᵗ, d, 1) 
        for yᵗ in 1:nstates(U, d + 1), y2 in 1:nstates(U,1))
end


#####################################################

prob_y_dummy(wᵢ::U, xᵢᵗ⁺¹, xᵢᵗ, xₖᵗ, y1, d, j) where U = prob_y(wᵢ::U, xᵢᵗ⁺¹, xᵢᵗ, y1, d) 


function f_bp_partial_ij(A::MPEM2, wᵢ::Vector{U}, ϕᵢ, d::Integer, qj, j) where {U<:RecursiveBPFactor}
    _f_bp_partial(A, wᵢ, ϕᵢ, d, prob_y_partial, qj, j)
end

function f_bp_partial_i(A::MPEM2, wᵢ::Vector{U}, ϕᵢ, d::Integer) where {U<:RecursiveBPFactor}
    _f_bp_partial(A, wᵢ, ϕᵢ, d, prob_y_dummy, 1, 1)
end

function _f_bp_partial(A::MPEM2, wᵢ::Vector{U}, ϕᵢ, 
    d::Integer, prob::Function, qj, j) where {U<:RecursiveBPFactor}
    q = length(ϕᵢ[1])
    B = [zeros(size(a,1), size(a,2), q, qj, q) for a in A]
    for t in 1:getT(A)
        Aᵗ,Bᵗ = A[t], B[t]
        @tullio Bᵗ[m,n,xᵢᵗ,xⱼᵗ,xᵢᵗ⁺¹] = prob(wᵢ[$t],xᵢᵗ⁺¹,xᵢᵗ,xⱼᵗ,yᵗ,d,j)*Aᵗ[m,n,yᵗ,xᵢᵗ]*ϕᵢ[$t][xᵢᵗ]
    end
    Aᵀ,Bᵀ = A[end], B[end]
    @tullio Bᵀ[m,n,xᵢᵀ,xⱼᵀ,xᵢᵀ⁺¹] = Aᵀ[m,n,yᵀ,xᵢᵀ] * ϕᵢ[end][xᵢᵀ]
    any(any(isnan, b) for b in B) && println("NaN in tensor train")
    return MPEM3(B)
end

function compute_prob_ys(wᵢ::Vector{U}, qi::Int, μin::Vector{<:MPEM2}, ψout, T, svd_trunc;
        svd_verbose::Bool=false) where {U<:RecursiveBPFactor}
    @assert all(normalization(a) ≈ 1 for a in μin)
    yrange = Base.OneTo(nstates(U, 1))
    B = map(eachindex(ψout)) do k
        Bk = map(zip(wᵢ, μin[k], ψout[k])) do (wᵢᵗ, μₖᵢᵗ, ψᵢₖᵗ)
            @tullio _[m,n,yₖ,xᵢ] := prob_xy(wᵢᵗ,yₖ,xₖ,xᵢ,k) * μₖᵢᵗ[m,n,xₖ,xᵢ] * ψᵢₖᵗ[xᵢ,xₖ] (yₖ in yrange)
        end |> MPEM2
        Bk, 0.0, 1
    end

    function op((B1, lz1, d1), (B2, lz2, d2))
        yrange = Base.OneTo(nstates(U,d1+d2))
        B = map(zip(wᵢ,B1,B2)) do (wᵢᵗ,B₁ᵗ,B₂ᵗ)
            @tullio B3[m1,m2,n1,n2,y,xᵢ] := prob_yy(wᵢᵗ,y,y1,y2,xᵢ,d1,d2) * B₁ᵗ[m1,n1,y1,xᵢ] * B₂ᵗ[m2,n2,y2,xᵢ] (y in yrange)
            @cast _[(m1,m2),(n1,n2),y,xᵢ] := B3[m1,m2,n1,n2,y,xᵢ]
        end |> MPEM2
        lz = normalize!(B)
        sweep_LtoR!(B, svd_trunc=TruncThresh(0.0))
        sweep_RtoL!(B; svd_trunc, verbose=svd_verbose)
        B, lz + lz1 + lz2, d1 + d2
    end

    Minit = fill(1.0, 1, 1, 1, qi)
    init = (MPEM2(fill(Minit, T + 1)), 0.0, 0)
    dest, (full, )  = cavity(B, op, init)
    (C, logzs) = unzip(dest)
    logzᵢ = sum(logzs)
    C, full, logzᵢ
end


# compute outgoing messages from node `i`
function onebpiter!(bp::MPBP{G,F}, i::Integer, ::Type{U}; 
        svd_trunc::SVDTrunc=TruncThresh(1e-6), svd_verbose::Bool=false) where {G<:AbstractIndexedDiGraph,F<:Real,U<:RecursiveBPFactor}
    @unpack g, w, ϕ, ψ, μ = bp
    ein, eout = inedges(g,i), outedges(g, i)
    wᵢ, ϕᵢ, dᵢ  = w[i], ϕ[i], length(ein)
    @assert wᵢ[1] isa U
    C, full, logzᵢ = compute_prob_ys(wᵢ, nstates(bp,i), μ[ein.|>idx], ψ[eout.|>idx], getT(bp), svd_trunc)
    for (j,e) = enumerate(eout)
        B = f_bp_partial_ij(C[j], wᵢ, ϕᵢ, dᵢ - 1, nstates(bp, dst(e)), j)
        μj = sweep_RtoL!(mpem2(B); svd_trunc, verbose=svd_verbose)
        logzᵢ += normalize!(μj)
        μ[idx(e)] = μj
    end
    B = f_bp_partial_i(full, wᵢ, ϕᵢ, dᵢ)
    bp.b[i] = B |> mpem2 |> marginalize
    bp.f[i] = dᵢ == 0 ? 0.0 : -logzᵢ / dᵢ
    nothing
end

