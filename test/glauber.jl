using Graphs
include("../mpdbp.jl")

q = q_glauber
T = 3

J = [0 1 0 1 1;
     1 0 1 1 0;
     0 1 0 0 1;
     1 1 0 0 0;
     1 0 1 0 0] .|> float
gg = SimpleGraph(J)
h = ones(nv(gg))

w = map(1:nv(gg)) do i
    map(1:T) do t
        GlauberFactor([J[i,j] for j in neighbors(gg, i)], h[i])
    end
end

g = IndexedBiDiGraph(gg)


A = [ mpem2(q, T) for e in edges(g) ]

bp = mpdbp(g, w, q, T)

i = 1
j_index = rand(1:indegree(g,i))
j = neighbors(g,i)[j_index]
A = bp.μ[inedges(bp.g, i) .|> idx]
f_bp(A, bp.p⁰[i], bp.w[i], bp.ϕ[i], j_index)