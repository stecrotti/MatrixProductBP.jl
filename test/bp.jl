include("../bp.jl")

tensors1 = [rand(1,3,2,2), rand(3,4,2,2), rand(4,1,2,2)]
tensors2 = [rand(1,3,2,2), rand(3,4,2,2), rand(4,1,2,2)]
tensors3 = [rand(1,3,2,2), rand(3,4,2,2), rand(4,1,2,2)]
A = [MPEM2(tensors1), MPEM2(tensors2), MPEM2(tensors3)]
z = length(A) + 1

T = 2
pᵢ = [0.2, 0.8]
ϕᵢ = map(1:T) do t 
    r = rand()
    [r, 1-r]
end
wᵢ = map(1:T) do t 
    f(xᵗ⁺¹::Integer, xₙᵢ::Vector{<:Integer}, xᵗ::Integer) = rand()
end
j_index = 2

B = f_bp(A, pᵢ, wᵢ, ϕᵢ, j_index)