```
SVD truncator. Can be threshold-based or bond size-based
```
abstract type SVDTrunc; end


# print info about truncations
function _debug_svd(M, U, λ, V, mprime)
    msg = """M$(size(M))=U$(size(U))*Λ$((length(λ),length(λ)))*V$(size(V'))
    Truncation to $mprime singular values.
    Error=$(sum(abs2, λ[mprime+1:end]) / sum(abs2, λ) |> sqrt)"""
    @debug "svd: "*msg
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

# truncates matrices to size `mprime`, stores the maximum error
struct TruncBondMax <: SVDTrunc
    mprime :: Int
    maxerr :: Vector{Float64}
    TruncBondMax(mprime::Int) = new(mprime, [0.0])
end

function (svd_trunc::TruncBondMax)(M::AbstractMatrix)
    U, λ, V = svd(M)
    mprime = min(length(λ), svd_trunc.mprime)
    _debug_svd(M, U, λ, V, mprime)
    err = sum(abs2, @view λ[mprime+1:end]) / sum(abs2, λ) |> sqrt
    svd_trunc.maxerr[1] = max(svd_trunc.maxerr[1], err) 
    U[:,1:mprime], λ[1:mprime], V[:,1:mprime]
end

function Base.show(io::IO, svd_trunc::TruncBondMax)
    println(io, "SVD truncation to bond size m'=", svd_trunc.mprime,
        ". Max error ", only(svd_trunc.maxerr))
end