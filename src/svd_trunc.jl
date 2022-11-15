```
SVD truncator. Can be threshold-based or bond size-based
```
abstract type SVDTrunc; end

struct TruncThresh{T} <: SVDTrunc
    ε :: T
end
function (svd_trunc::TruncThresh)(λ::Vector{<:Real})
    λ_norm = norm(λ)
    findlast(λₖ > svd_trunc.ε*λ_norm for λₖ in λ)
end

function show(io::IO, svd_trunc::TruncThresh)
    println(io, "SVD truncation with threshold ε=", svd_trunc.ε)
end

struct TruncBond <: SVDTrunc
    mprime :: Int
end
function (svd_trunc::TruncBond)(λ::Vector{<:Real}) 
    min(length(λ), svd_trunc.mprime)
end

function show(io::IO, svd_trunc::TruncBond)
    println(io, "SVD truncation to bond size m'=", svd_trunc.mprime)
end