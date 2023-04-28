# construct a uniform mpem with given bond dimensions

MPEM2(tensors::Vector{Array{Float64,4}}) = MatrixProductTrain(tensors)

# construct a uniform mpem with given bond dimensions
mpem2(q1::Int, q2, T::Int; d::Int=2, bondsizes=[1; fill(d, T); 1]) = mpem(bondsizes, q1, q2)

# construct a uniform mpem with given bond dimensions
rand_mpem2(q1::Int, q2::Int, T::Int; d::Int=2, bondsizes=[1; fill(d, T); 1]) = rand_mpem(bondsizes, q1, q2)

function firstvar_marginal(A::MPEM2; p = marginals(A))
    map(p) do pₜ
        pᵢᵗ = sum(pₜ, dims=2) |> vec
        pᵢᵗ ./= sum(pᵢᵗ)
    end
end

function marginalize(A::MPEM2)
    MPEM1([@tullio b[m, n, xi] := a[m, n, xi, xj] for a in A])
end
