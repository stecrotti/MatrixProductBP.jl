#### Infinite Uniform MPEMS
const AbstractInfiniteUniformMPEM{F} = InfiniteUniformTensorTrain{F}
nstates(A::AbstractInfiniteUniformMPEM) = size(A.tensor, 3)

const InfiniteUniformMPEM1{F} = InfiniteUniformTensorTrain{F, 3}
flat_uniform_infinite_mpem1(q::Int; d::Int=2) = flat_infinite_uniform_tt(d, q)
rand_mpem1(q::Int; d::Int=2) = rand_infinite_uniform_tt(d, q)

const InfiniteUniformMPEM2{F} = InfiniteUniformTensorTrain{F, 4}
flat_uniform_infinite_mpem2(q1::Int, q2::Int; d::Int=2) = flat_infinite_uniform_tt(d, q1, q2)
rand_mpem2(q1::Int, q2::Int; d::Int=2) = rand_infinite_uniform_tt(d, q1, q2)

function marginalize(A::InfiniteUniformMPEM2{F}) where F
    @tullio B[m, n, xi] := A.tensor[m, n, xi, xj]
    InfiniteUniformMPEM1{F}(B; z = A.z)
end

struct InfiniteUniformMPEM3{F<:Real}
    tensor  :: Array{F,5}
    z       :: Logarithmic{F}
    function InfiniteUniformMPEM3(tensor::Array{F,5}; z::Logarithmic{F}=Logarithmic(one(F))) where {F<:Real}
        size(tensor, 1) == size(tensor, 2) || throw(ArgumentError("Matrix must be square"))
        size(tensor, 3) == size(tensor, 5) ||
            throw(ArgumentError("First and third variable indices should have matching ranges because they both represent `xᵢ`"))
    
        return new{F}(tensor, z)
    end
end

function mpem2(B::InfiniteUniformMPEM3{F}) where {F}
    Bt = B.tensor
    qᵢᵗ = size(Bt, 3); qⱼᵗ = size(Bt, 4); qᵢᵗ⁺¹ = size(Bt, 5)

    @cast M[(xᵢᵗ, xⱼᵗ, m), (n, xᵢᵗ⁺¹)] |= Bt[m, n, xᵢᵗ, xⱼᵗ, xᵢᵗ⁺¹]
    U, λ, V = svd(M)   
    m = length(λ) 
    @cast C[m, k, xᵢᵗ, xⱼᵗ] := U[(xᵢᵗ, xⱼᵗ, m), k] k∈1:m, xᵢᵗ∈1:qᵢᵗ, xⱼᵗ∈1:qⱼᵗ
    @cast Vt[m, n, xᵢᵗ⁺¹] := V'[m, (n, xᵢᵗ⁺¹)]  xᵢᵗ⁺¹∈1:qᵢᵗ⁺¹
    @tullio D[m, n, xᵢᵗ, xⱼᵗ] := λ[m] * Vt[m, l, xᵢᵗ] * C[l, n, xᵢᵗ, xⱼᵗ]
    return InfiniteUniformMPEM2{F}(D; z = B.z)
end

mpem3from2(::Type{InfiniteUniformMPEM2{F}}) where F = InfiniteUniformMPEM3


#### Naive MPBP

function f_bp(A::Vector{M2}, wᵢ::Vector{U}, ϕᵢ::Vector{Vector{F}}, 
    ψₙᵢ::Vector{Vector{Matrix{F}}}, j_index::Integer; showprogress=false, 
    svd_trunc::SVDTrunc=TruncThresh(0.0), periodic=false) where {F,U<:BPFactor,M2<:InfiniteUniformMPEM2}

    @assert length(wᵢ) == 1
    @assert length(ϕᵢ) == 1
    q = length(ϕᵢ[1])
    qj = size(ψₙᵢ[j_index][1],1)
    @assert all(length(ϕᵢᵗ) == q for ϕᵢᵗ in ϕᵢ)
    @assert j_index in eachindex(A)
    notj = eachindex(A)[Not(j_index)]
    xin = Iterators.product((axes(ψₙᵢ[k][1],2) for k in notj)...)

    B = zeros(reduce(.*, (size(A[k].tensor)[1:2] for k in notj); init=(1,1))..., q, qj, q)
    @inbounds for xᵢᵗ in 1:q 
        for xₙᵢ₋ⱼᵗ in xin
            @views Aᵗ = kron(ones(1,1),ones(1,1),
                (A[k].tensor[:,:,xₖᵗ,xᵢᵗ] .* ψₙᵢ[k][1][xᵢᵗ, xₖᵗ] for (k, xₖᵗ) in zip(notj,xₙᵢ₋ⱼᵗ))...)
            for xⱼᵗ in 1:qj, xᵢᵗ⁺¹ in 1:q
                w = ϕᵢ[1][xᵢᵗ]
                xₙᵢᵗ = [xₙᵢ₋ⱼᵗ[1:j_index-1]..., xⱼᵗ, xₙᵢ₋ⱼᵗ[j_index:end]...]
                w *= wᵢ[1](xᵢᵗ⁺¹, xₙᵢᵗ, xᵢᵗ)
                if !iszero(w)
                    B[:, :, xᵢᵗ, xⱼᵗ, xᵢᵗ⁺¹] .+= Aᵗ .* w
                end
            end
        end
    end

    any(isnan, B) && @error "NaN in tensor"
    return mpem3from2(eltype(A))(B), zero(F)
end

function f_bp_dummy_neighbor(A::Vector{M2}, 
        wᵢ::Vector{U}, ϕᵢ::Vector{Vector{F}}, ψₙᵢ::Vector{Vector{Matrix{F}}};
        showprogress=false, svd_trunc::SVDTrunc=TruncThresh(0.0), periodic=false) where {F,U<:BPFactor,M2<:InfiniteUniformMPEM2}
    @assert length(wᵢ) == 1
    @assert length(ϕᵢ) == 1
    q = length(ϕᵢ[1])
    xin = Iterators.product((axes(ψₙᵢ[k][1],2) for k in eachindex(A))...)
    
    B = zeros(reduce(.*, (size(A[k].tensor)[1:2] for k in eachindex(A)); init=(1,1))..., q, 1, q)
    @inbounds for xᵢᵗ in 1:q
        for xₙᵢᵗ in xin
            @views Aᵗ = kron(ones(1,1),ones(1,1),
                (A[k].tensor[:,:,xₖᵗ,xᵢᵗ] .* ψₙᵢ[k][1][xᵢᵗ, xₖᵗ] for (k, xₖᵗ) in pairs(xₙᵢᵗ))...)
            for xᵢᵗ⁺¹ in 1:q
                w = ϕᵢ[1][xᵢᵗ]
                w *= wᵢ[1](xᵢᵗ⁺¹, collect(xₙᵢᵗ), xᵢᵗ)
                if !iszero(w)
                    B[:, :, xᵢᵗ, 1, xᵢᵗ⁺¹] .+= Aᵗ .* w
                end
            end
        end
    end
    
    any(isnan, B) && @error "NaN in tensor"
    return mpem3from2(eltype(A))(B), 0.0
end

function pair_belief_as_mpem(Aᵢⱼ::M2, Aⱼᵢ::M2, ψᵢⱼ) where {M2<:InfiniteUniformMPEM2}
    @cast A[(aᵗ,bᵗ),(aᵗ⁺¹,bᵗ⁺¹),xᵢᵗ,xⱼᵗ] := Aᵢⱼ.tensor[aᵗ,aᵗ⁺¹,xᵢᵗ, xⱼᵗ] * 
        Aⱼᵢ.tensor[bᵗ,bᵗ⁺¹,xⱼᵗ,xᵢᵗ] * only(ψᵢⱼ)[xᵢᵗ, xⱼᵗ]
    return M2(A)
end

function pair_belief(Aᵢⱼ::M2, Aⱼᵢ::M2, ψᵢⱼ) where {M2<:InfiniteUniformMPEM2}
    A = pair_belief_as_mpem(Aᵢⱼ, Aⱼᵢ, ψᵢⱼ)
    return marginals(A), normalization(A)
end


#### Recursive MPBP

function _f_bp_partial(A::InfiniteUniformMPEM2, wᵢ::Vector{U}, ϕᵢ, 
        d::Integer, prob::Function, qj, j) where {U<:RecursiveBPFactor}
    q = length(ϕᵢ[1])
    At = A.tensor
    B = zeros(size(At, 1), size(At, 2), q, qj, q)
    W = zeros(q, q, qj, size(At, 3))
    @tullio avx=false W[xᵢᵗ⁺¹,xᵢᵗ,xⱼᵗ,yᵗ] = prob(wᵢ[1],xᵢᵗ⁺¹,xᵢᵗ,xⱼᵗ,yᵗ,d,j)*ϕᵢ[1][xᵢᵗ]
    @tullio B[m,n,xᵢᵗ,xⱼᵗ,xᵢᵗ⁺¹] = W[xᵢᵗ⁺¹,xᵢᵗ,xⱼᵗ,yᵗ] * At[m,n,yᵗ,xᵢᵗ]
    any(isnan, B) && @error "NaN in tensor train"
    return InfiniteUniformMPEM3(B; z = A.z)
end

function compute_prob_ys(wᵢ::Vector{U}, qi::Int, μin::Vector{M2}, ψout, T, svd_trunc) where {U<:RecursiveBPFactor, M2<:InfiniteUniformMPEM2}
    @debug @assert all(float(normalization(a)) ≈ 1 for a in μin) "$([float(normalization(a)) for a in μin])"
    
    # Compute ̃m{j→i}(̅y_{j},̅xᵢ) for j∈∂i
    B = map(eachindex(ψout)) do k
        wᵢᵗ, μₖᵢᵗ, ψᵢₖᵗ = wᵢ[1], μin[k][1], ψout[k][1]
        Pxy = zeros(nstates(wᵢᵗ,1), size(μₖᵢᵗ, 3), qi)
        @tullio avx=false Pxy[yₖ,xₖ,xᵢ] = prob_xy(wᵢᵗ,yₖ,xₖ,xᵢ,k) * ψᵢₖᵗ[xᵢ,xₖ]
        @tullio Bₖ[m,n,yₖ,xᵢ] := Pxy[yₖ,xₖ,xᵢ] * μₖᵢᵗ[m,n,xₖ,xᵢ] 
        M2(Bₖ), 1
    end

    # the operation that combines together two ̃m messages
    function op((B1, d1), (B2, d2))
        B1t = B1.tensor; B2t = B2.tensor
        wᵢᵗ = wᵢ[1]
        Pyy = zeros(nstates(wᵢᵗ,d1+d2), size(B1t,3), size(B2t,3), size(B1t,4))
        @tullio avx=false Pyy[y,y1,y2,xᵢ] = prob_yy(wᵢᵗ,y,y1,y2,xᵢ,d1,d2) 
        @tullio B3[m1,m2,n1,n2,y,xᵢ] := Pyy[y,y1,y2,xᵢ] * B1t[m1,n1,y1,xᵢ] * B2t[m2,n2,y2,xᵢ]
        @cast BB[(m1,m2),(n1,n2),y,xᵢ] := B3[m1,m2,n1,n2,y,xᵢ]
        
        Bout = M2(BB; z = B1.z * B2.z)
        any(isnan, Bout.tensor) && @error "NaN in tensor train"
        # compress!(Bout; svd_trunc)
        normalize_eachmatrix!(Bout)    # keep this one?
        any(isnan, Bout.tensor) && @error "NaN in tensor train"
        Bout, d1 + d2
    end

    Minit = [float(prob_y0(wᵢ[1], y, xᵢ)) for _ in 1:1,
                _ in 1:1,
                y in 1:nstates(wᵢ[1],0),
                xᵢ in 1:qi]
    init = (M2(Minit), 0)
    # compute all-but-one `op`s
    dest, (full,) = cavity(B, op, init)
    (C,) = unzip(dest)
    C, full, B
end

function set_msg!(bp::MPBP{G,F,V,M2}, μj::M2, edge_id, damp, svd_trunc) where {G,F,V,M2<:InfiniteUniformMPEM2}
    @assert 0 ≤ damp < 1
    logzᵢ₂ⱼ = normalize!(μj)
    if damp > 0 
        @warn "Damping not yet implemented for infinite messages"
    end
    bp.μ[edge_id] = μj
    logzᵢ₂ⱼ
end

#### Misc

function mpbp_stationary(g::IndexedBiDiGraph{Int}, w::Vector{<:Vector{<:BPFactor}},
    q::AbstractVector{Int}; 
    d::Int=1,
    ϕ = [[ones(q[i]) for t in 1:1] for i in vertices(g)],
    ψ = [[ones(q[i],q[j]) for t in 1:1] for (i,j) in edges(g)],
    μ = [flat_uniform_infinite_mpem2(q[i], q[j]; d) for (i,j) in edges(g)],
    b = [flat_uniform_infinite_mpem1(q[i]; d) for i in vertices(g)],
    f = zeros(nv(g)))

    @assert all(length(wᵢ) == 1 for wᵢ in w)
    return MPBP(g, w, ϕ, ψ, μ, b, f)
end

const MPBPStationary = MPBP{<:AbstractIndexedDiGraph, <:Real, <:AbstractVector{<:BPFactor},<:InfiniteUniformMPEM2,<:InfiniteUniformMPEM1} where {G,F,V}

is_periodic(bp::MPBPStationary) = true

function means(f, bp::MPBP{G,F,V,M2}; sites=vertices(bp.g)) where {G,F,V,M2<:InfiniteUniformMPEM2}
    map(sites) do i
        expectation.(x->f(x, i), real(marginals(bp.b[i])))
    end
end

function mpbp_stationary_infinite_graph(k::Integer, wᵢ::Vector{U}, qi::Int,
    ϕᵢ = fill(ones(qi), length(wᵢ));
    ψₖᵢ = fill(ones(qi, qi), length(wᵢ)),
    d::Int=1) where {U<:BPFactor}

    T = length(wᵢ) - 1
    @assert T == 0
    @assert length(ϕᵢ) == T + 1
    @assert length(ψₖᵢ) == T + 1
    
    g = InfiniteRegularGraph(k)
    μ = flat_uniform_infinite_mpem2(qi, qi; d)
    b = flat_uniform_infinite_mpem1(qi; d)
    MPBP(g, [wᵢ], [ϕᵢ], [ψₖᵢ], [μ], [b], [0.0])
end

function mpbp_stationary_infinite_bipartite_graph(k::NTuple{2,Int}, wᵢ::Vector{Vector{U}},
    qi::NTuple{2,Int},
    ϕᵢ = [fill(ones(qi[i]), length(wᵢ[1])) for i in 1:2];
    ψₖᵢ = [fill(ones(qi[i], qi[3-i]), length(wᵢ[1])) for i in 1:2],
    d=(1, 1)) where {U<:BPFactor}

    T = length(wᵢ[1]) - 1
    @assert T == 0
    @assert length(wᵢ[2]) == T + 1
    @assert all(isequal(T+1), length.(ϕᵢ))
    @assert all(isequal(T+1), length.(ψₖᵢ))

    g = InfiniteBipartiteRegularGraph(k)
    μ = [flat_uniform_infinite_mpem2(qi[i], qi[3-i]; d=d[i]) for i in 1:2]
    b = [flat_uniform_infinite_mpem1(qi[i]; d=d[i]) for i in 1:2]
    MPBP(g, wᵢ, ϕᵢ, ψₖᵢ, μ, b, zeros(2))
end

function reset_messages!(bp::MPBPStationary)
    for A in bp.μ
        A.tensor .= 1
        normalize!(A)
    end
    return nothing
end
function reset_beliefs!(bp::MPBPStationary)
    for A in bp.b
        A.tensor .= 1
        normalize!(A)
    end
    return nothing
end

default_truncator(::Type{<:InfiniteUniformMPEM2}) = TruncVUMPS(4)
    
struct CB_BPVUMPS{TP<:ProgressUnknown, F, M2<:InfiniteUniformMPEM2}
    prog :: TP
    m    :: Vector{Vector{Vector{Float64}}} 
    Δs   :: Vector{Float64}     # convergence error on marginals
    A    :: Vector{Vector{M2}}     
    εs   :: Vector{Float64}     # convergence error on messages
    f    :: F

    function CB_BPVUMPS(bp::MPBPStationary{G, T, V, M2}; showprogress::Bool=true, f::F=(x,i)->x, info="") where {G, T, V, M2, F}
        dt = showprogress ? 0.1 : Inf
        isempty(info) || (info *= "\n")
        prog = ProgressUnknown(desc=info*"Running MPBP: iter", dt=dt, showspeed=true)
        TP = typeof(prog)
        m = [means(f, bp)]
        Δs = zeros(0)
        A = [deepcopy(bp.μ.v)]
        εs = zeros(0)
        new{TP,F,M2}(prog, m, Δs, A, εs, f)
    end
end

function (cb::CB_BPVUMPS)(bp::MPBPStationary, it::Integer, svd_trunc::SVDTrunc)
    marg_new = means(cb.f, bp)
    marg_old = cb.m[end]
    Δ = isempty(marg_new) ? NaN : maximum(maximum(abs, mn .- mo) for (mn, mo) in zip(marg_new, marg_old))
    push!(cb.Δs, Δ)
    push!(cb.m, marg_new)
    A_new = bp.μ
    A_old = cb.A[end]
    ε = isempty(A_new) ? NaN : maximum(abs, 1 - dot(Anew, Aold) for (Anew, Aold) in zip(A_new, A_old))
    push!(cb.εs, ε)
    push!(cb.A, deepcopy(bp.μ))
    next!(cb.prog, showvalues=[(:Δ,Δ), (:trunc, summary_compact(svd_trunc))])
    flush(stdout)
    return Δ
end