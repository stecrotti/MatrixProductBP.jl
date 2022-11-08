include("../sis.jl")
include("../bp_fast.jl")

q = q_sis
T = 1

A = [0 1 0 0;
     1 0 1 0;
     0 1 0 1;
     0 0 1 0]
g = IndexedGraph(A)
N = 4

λ = 0.2
κ = 0.1 

p⁰ = map(1:N) do i
    r = 0.95
    [r, 1-r]
end
ϕ = [[ones(2) for t in 1:T] for i in 1:N]
ψ = [[ones(2,2) for t in 1:T] for _ in 1:2*ne(g)]

sis = SIS(g, λ, κ, p⁰, ϕ, ψ)
bp = mpdbp(sis)
svd_trunc = TruncThresh(0.0)

i = 3
@unpack g, w, ϕ, ψ, p⁰, μ = bp
ein = inedges(g,i)
eout = outedges(g, i)
A = μ[ein.|>idx]
j = 1; e_out = first(eout)
pᵢ⁰ = p⁰[i]; wᵢ = w[i]; ϕᵢ = ϕ[i]; ψₙᵢ = ψ[eout.|>idx]
B = f_bp(A, pᵢ⁰, wᵢ, ϕᵢ, ψₙᵢ, j)
C = mpem2(B); D = sweep_RtoL!(C; svd_trunc)
B_fast = f_bp_fast(A, pᵢ⁰, wᵢ, ϕᵢ, ψₙᵢ, j; svd_trunc)
C_fast = mpem2(B_fast); D_fast = sweep_RtoL!(C_fast; svd_trunc)

p_fast = evaluate(D_fast, fill([S,S], T+1))
p = evaluate(D, fill([S,S], T+1))
p_fast, p


# function onebpiter_fast!(bp::MPdBP, i::Integer; svd_trunc::SVDTrunc=TruncThresh(1e-6))
#     @unpack g, w, ϕ, ψ, p⁰, μ = bp
#     ein = inedges(g,i)
#     eout = outedges(g, i)
#     A = μ[ein.|>idx]
#     @assert all(normalization(a) ≈ 1 for a in A)
#     zᵢ = 1.0
#     for (j_ind, e_out) in enumerate(eout)
#         B = f_bp_fast(A, p⁰[i], w[i], ϕ[i], ψ[eout.|>idx], j_ind; svd_trunc)
#         C = mpem2(B)
#         μ[idx(e_out)] = sweep_RtoL!(C; svd_trunc)
#         zᵢ₂ⱼ = normalize!(μ[idx(e_out)])
#         zᵢ *= zᵢ₂ⱼ
#     end
#     dᵢ = length(ein)
#     return zᵢ ^ (1 / dᵢ)
# end

# i = 3
# bp_fast = deepcopy(bp)
# onebpiter!(bp, i; svd_trunc)
