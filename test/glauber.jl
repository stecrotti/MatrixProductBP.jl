using Graphs
include("../mpdbp.jl")

# q = q_glauber
# T = 4
# ε = 1e-6

# J = [0 1 0;
#      1 0 1;
#      0 1 0] .|> float
# N = 3
# h = zeros(N)
# β = 1.0

# p⁰ = map(1:N) do i
#     # r = rand()
#     r = 0.5
#     [r, 1-r]
# end
# ϕ = [[[0.5,0.5] for t in 1:T] for i in 1:N]

# ising = Ising(J, h, β)
# gl = ExactGlauber(ising, p⁰, ϕ)
# m = site_marginals(gl)
# mm = site_time_marginals(gl; m)

bp = mpdbp(gl)
cb = CB_BP(bp)
iterate!(bp, maxiter=15, ε=0.0; cb)

b = beliefs(bp, ε=0.0)