include("../sis.jl")
include("sib.jl")
include("../exact/montecarlo.jl")
include("../exact/exact.jl")

q = q_sis
T = 2

A = [0 1 0 0;
     1 0 1 0;
     0 1 0 1;
     0 0 1 0]
g = IndexedGraph(A)
N = 4

λ = 0.2
κ = 0.0 # SI

p⁰ = map(1:N) do i
    r = 0.95
    [r, 1-r]
end
ϕ = [[ones(2) for t in 1:T] for i in 1:N]
ψ = [[ones(2,2) for t in 1:T] for _ in 1:2*ne(g)]

sis = SIS(g, λ, κ, p⁰, ϕ, ψ)
bp = mpdbp(sis)
draw_node_observations!(bp, N, last_time=true)

b_sib, F_sib, F_sib_detail = sib_SI(T, g, bp.ϕ, p⁰, λ)
p_sib = [[bbb[2] for bbb in bb] for bb in b_sib]

svd_trunc = TruncThresh(1e-5)
# svd_trunc = TruncBond(2)

cb = CB_BP(bp)
iterate!(bp, maxiter=10; svd_trunc, cb)

b_bp = beliefs(bp)
p_bp =  [[bbb[2] for bbb in bb] for bb in b_bp]

# sms = sample(bp, 10^5)
# b_mc = marginals(sms)
# p_mc = [[bbb[2] for bbb in bb] for bb in b_mc]

# [ p_bp[1][1:end-1] p_sib[1] p_mc[1][1:end-1] ]

p_exact, Z_exact = exact_prob(bp)

F_exact = -log(Z_exact)
F_bp_detail = free_energy_factors(bp)
F_bp = sum(F_bp_detail)
@show F_exact F_sib F_bp;