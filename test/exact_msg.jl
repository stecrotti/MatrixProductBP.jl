using MatrixProductBP, Test
import MatrixProductBP: f_bp, eachstate, zero_exact_msg
using Random, MatrixProductBP.Models, Graphs, IndexedGraphs

# @testset "Exact messages" begin
    rng = MersenneTwister(111)

    T = 2
    J = [0 1 0 0 0;
        1 0 1 1 0;
        0 1 0 0 0;
        0 1 0 0 0;
        0 0 0 0 0] .|> float

    N = size(J, 1)
    h = randn(rng, N)

    β = 1.0
    ising = Ising(J, h, β)

    gl = Glauber(ising, T)

    for i in 1:N
        r = 0.75
        gl.ϕ[i][1] .*= [r, 1-r]
    end

    bp = mpbp(deepcopy(gl))
    bp_ex = mpbp_exact(bp.g, bp.w, fill(2, nv(bp.g)), T; ϕ = bp.ϕ, ψ = bp.ψ)
    @test bp_ex isa MatrixProductBP.MPBPExact

    iterate!(bp_ex; maxiter=20)
    svd_trunc = TruncThresh(0.0)
    iterate!(bp; maxiter=20, svd_trunc)
    @test beliefs(bp) ≈ beliefs(bp_ex)
    @test bethe_free_energy(bp) ≈ bethe_free_energy(bp)

    f(x,i) = 2x-3
    autocorrelations(f, bp_ex)
# end