using Test

include("../mpem.jl")

@testset "MPEM2" begin
    tensors = [rand(1,3,2,2), rand(3,4,2,2), rand(4,10,2,2), rand(10,1,2,2)]
    C = MPEM2(tensors)
    T = getT(C)
    x = [rand(1:2,2) for t in 1:T+1]
    e1 = evaluate(C, x)
    n1 = norm(C)

    sweep_RtoL!(C; ε=0.0)
    e2 = evaluate(C, x)
    @test e2 ≈ e1
    n2 = norm(C)
    @test n2 ≈ n1
    n2_fast = norm_fast_R(C)
    @test n2_fast ≈ n1

    sweep_LtoR!(C, ε=0.0)
    e3 = evaluate(C, x)
    @test e3 ≈ e1
    n3 = norm(C)
    @test n3 ≈ n1
    n3_fast = norm_fast_L(C)
    @test n3_fast ≈ n1
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


tensors = [rand(1,3,2,2), rand(3,4,2,2), rand(4,10,2,2), rand(10,1,2,2)]
C = MPEM2(tensors)
@show bond_dims(C)
T = getT(C)
x = [rand(1:2,2) for t in 1:T+1]
e1 = evaluate(C, x)
n1 = norm(C)

sweep_RtoL!(C; ε=0.2)
@show bond_dims(C)
e2 = evaluate(C, x)
@show e1, e2
n2 = norm(C)
@show n1, n2
n2_fast = norm_fast_R(C)
@test n2_fast ≈ norm(C)

# sweep_LtoR!(C, ε=0.0)
# e3 = evaluate(C, x)
# @test e3 ≈ e1
# n3 = norm(C)
# @test n3 ≈ n1
# n3_fast = norm_fast_L(C)
# @test n3_fast ≈ n1