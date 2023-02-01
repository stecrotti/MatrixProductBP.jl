import MatrixProductBP: sweep_RtoL!, sweep_LtoR!

svd_trunc = TruncThresh(0.0)

@testset "MPEM2" begin
    tensors = [rand(1,3,2,2), rand(3,4,2,2), rand(4,10,2,2), rand(10,1,2,2)]
    C = MPEM2(tensors)
    T = getT(C)
    x = [rand(1:2,2) for t in 1:T+1]
    e1 = evaluate(C, x)

    sweep_RtoL!(C; svd_trunc)
    e2 = evaluate(C, x)
    @test e2 ≈ e1

    sweep_LtoR!(C; svd_trunc)
    e3 = evaluate(C, x)
    @test e3 ≈ e1

    @test [sum(x,dims=2) for x in pair_marginal(C)] ≈ firstvar_marginal(C)
end

@testset "MPEM2 random" begin
    T = 5
    q1 = 2; q2 = 4
    C = rand_mpem2(q1, q2, T)
    x = [[rand(1:q1), rand(1:q2)] for t in 1:T+1]
    e1 = evaluate(C, x)

    sweep_RtoL!(C; svd_trunc)
    e2 = evaluate(C, x)
    @test e2 ≈ e1

    sweep_LtoR!(C; svd_trunc)
    e3 = evaluate(C, x)
    @test e3 ≈ e1

    @test [sum(x,dims=2) for x in pair_marginal(C)] ≈ firstvar_marginal(C)
end

@testset "MPEM3" begin
    tensors = [ rand(1,3,2,2,2), rand(3,4,2,2,2), rand(4,1,2,2,2) ]
    tensors[end][:,:,:,:,2] .= tensors[end][:,:,:,:,1]
    B = MPEM3(tensors)
    T = getT(B)

    x = [rand(1:2,2) for t in 1:T+1]
    e1 = evaluate(B, x)

    C = mpem2(B)
    e2 = evaluate(C,x)
    @test e2 ≈ e1
end

@testset "Accumulators" begin
    tensors = [rand(1,3,2,2), rand(3,4,2,2), rand(4,10,2,2), rand(10,1,2,2)]
    A = MPEM2(tensors)
    L = MatrixProductBP.MPEMs.accumulate_L(A)
    R = MatrixProductBP.MPEMs.accumulate_R(A)
    @test L[end] ≈ R[begin]
end

@testset "Sum of MPEMs" begin
    for N in 1:3
        for q in 1:3
            qs = fill(q, N)
            T = 5
            A = MatrixProductBP.MPEMs.rand_mpem( [1; rand(1:7, T-2); 1], qs... )
            B = MatrixProductBP.MPEMs.rand_mpem( [1; rand(1:7, T-2); 1], qs... )
            x = [rand(1:q[1],N) for t in 0:T]
            @test evaluate(A, x) + evaluate(B, x) ≈ evaluate(A+B, x)
        end
    end
end