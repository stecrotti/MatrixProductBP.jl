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
end

@testset "MPEM3" begin
    tensors = [ rand(2,2,1,3,2), rand(2,2,3,4,2), rand(2,2,4,1,2) ]
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