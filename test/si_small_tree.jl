include("../sis.jl")
include("sib.jl")

q = q_sis
T = 6

A = [0 1 0 0 0;
     1 0 1 0 0;
     0 1 0 1 1;
     0 0 1 0 0;
     0 0 1 0 0]
g = IndexedGraph(A)
N = 5

λ = 0.1
κ = 0.0 # SI

p⁰ = map(1:N) do i
    r = rand()
    r = 0.95
    [r, 1-r]
end
ϕ = [[ones(2) for t in 1:T] for i in 1:N]
ϕ[1][1] = [1, 0]
ϕ[2][2] = [0, 1]
ϕ[3][3] = [0, 1]

b_sib = sib_SI(T, g, ϕ, p⁰, λ)
p_sib = [[bbb[2] for bbb in bb] for bb in b_sib]

sis = SIS(g, λ, κ, p⁰, ϕ)

svd_trunc = TruncThresh(1e-5)
# svd_trunc = TruncBond(2)
bp = mpdbp(sis)
cb = CB_BP(bp)
iterate!(bp, maxiter=10; svd_trunc, cb)

b_bp = beliefs(bp; svd_trunc)
p_bp =  [[bbb[2] for bbb in bb] for bb in b_bp]

p_bp[1][1:end-1]
p_sib[1]
