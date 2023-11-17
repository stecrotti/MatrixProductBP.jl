@testset "Exact messages" begin
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
    bp_ex = mpbp_exact(bp.g, bp.w, fill(2, nv(bp.g)), T)
    @test bp_ex isa MatrixProductBP.MPBPExact


end