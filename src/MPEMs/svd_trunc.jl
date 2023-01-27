```
SVD truncator. Can be threshold-based or bond size-based
```
abstract type SVDTrunc; end


# print info about truncations
function _debug_svd(M, U, λ, V, mprime)
    @debug "svd" """M$(size(M))=U$(size(U))*Λ$((length(λ),length(λ)))*V$(size(V'))
    Truncation to $mprime singular values.
    Error=$(sum(abs2, λ[mprime+1:end]) / sum(abs2, λ) |> sqrt)"""
end

struct TruncThresh{T} <: SVDTrunc
    ε :: T
end
function (svd_trunc::TruncThresh)(M::AbstractMatrix)
    U, λ, V = svd(M)
    λ_norm = norm(λ)
    mprime = findlast(λₖ > svd_trunc.ε*λ_norm for λₖ in λ)
    _debug_svd(M, U, λ, V, mprime)
    U[:,1:mprime], λ[1:mprime], V[:,1:mprime]
end

function show(io::IO, svd_trunc::TruncThresh)
    println(io, "SVD truncation with threshold ε=", svd_trunc.ε)
end

struct TruncBond <: SVDTrunc
    mprime :: Int
end
function (svd_trunc::TruncBond)(M::AbstractMatrix)
    U, λ, V = svd(M)
    mprime = min(length(λ), svd_trunc.mprime)
    _debug_svd(M, U, λ, V, mprime)
    U[:,1:mprime], λ[1:mprime], V[:,1:mprime]
end

function show(io::IO, svd_trunc::TruncBond)
    println(io, "SVD truncation to bond size m'=", svd_trunc.mprime)
end