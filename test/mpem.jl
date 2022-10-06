include("../mpem.jl")

# tensors = [rand(1,3,2,2), rand(3,4,2,2), rand(4,10,2,2), rand(10,1,2,2)]
# # tensors = [rand(1,3,2,2), rand(3,1,2,2)]
# C = MPEM2(tensors)
# T = getT(C)

# x = [rand(1:2,2) for t in 1:T+1]
# @show evaluate(C, x)
# @show norm(C)

# sweep_RtoL!(C; ε=0.0)
# @show evaluate(C, x)
# @show norm(C)
# # @show bond_dims(C)

# sweep_LtoR!(C, ε=0.0)
# # @show bond_dims(C)
# @show evaluate(C, x)
# @show norm(C)

# sweep_RtoL!(C; ε=0.0)
# @show evaluate(C, x)
# @show norm(C)


tensors = [ rand(2,2,1,3,2), rand(2,2,3,4,2), rand(2,2,4,1,2) ]
tensors[end][:,:,:,:,2] .= tensors[end][:,:,:,:,1]
B = MPEM3(tensors)
T = getT(B)

x = [rand(1:2,2) for t in 1:T+1]
@show evaluate(B, x)

C = mpem2(B)
@show evaluate(C,x)
