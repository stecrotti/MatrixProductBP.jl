import MatrixProductBP: rand_mpem1, rand_mpem2

@testset "MPEM1" begin
    q = 2
    T = 10
    d = 4
    A = rand_mpem1(q, T; d)
    x = [rand(1:q,1) for _ in A]
    e1 = evaluate(A, x)

    orthogonalize_left!(A)
    e2 = evaluate(A, x)
    @test e2 ≈ e1
end

@testset "MPEM2" begin
    q = 2
    T = 5
    d = 4
    A = rand_mpem2(q, q, T; d)
    x = [rand(1:q,2) for _ in A]
    e1 = evaluate(A, x)

    orthogonalize_left!(A)
    e2 = evaluate(A, x)
    @test e2 ≈ e1
end

@testset "MPEM3" begin
    tensors = [ rand(1,3,2,2,2), rand(3,4,2,2,2), rand(4,1,2,2,2) ]
    tensors[end][:,:,:,:,2] .= tensors[end][:,:,:,:,1]
    B = MPEM3(tensors)

    x = [rand(1:2,2) for _ in B]
    e1 = evaluate(B, x)

    C = mpem2(B)
    e2 = evaluate(C,x)
    @test e2 ≈ e1
end

@testset "periodic MPEM2" begin
    q = 2
    T = 5
    d = 4
    A = rand_periodic_tt(d, T+1, 2, 2)
    x = [rand(1:q,2) for _ in A]
    e1 = evaluate(A, x)

    orthogonalize_left!(A)
    e2 = evaluate(A, x)
    @test e2 ≈ e1
end

@testset "periodic MPEM3" begin
    tensors = [ rand(1,3,2,2,2), rand(3,4,2,2,2), rand(4,1,2,2,2) ]
    B = PeriodicMPEM3(tensors)

    x = [rand(1:2,2) for _ in B]
    e1 = evaluate(B, x)

    C = mpem2(B)
    e2 = evaluate(C,x)
    @test e2 ≈ e1
end
