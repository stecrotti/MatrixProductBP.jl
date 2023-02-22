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

summary(::SVDTrunc) = error("Not implemented")
show(io::IO, svd_trunc::SVDTrunc) = println(io, summary(svd_trunc))

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

summary(svd_trunc::TruncThresh) = "SVD truncation with threshold ε="*string(svd_trunc.ε)

struct TruncBond <: SVDTrunc
    mprime :: Int
end
function (svd_trunc::TruncBond)(M::AbstractMatrix)
    U, λ, V = svd(M)
    mprime = min(length(λ), svd_trunc.mprime)
    _debug_svd(M, U, λ, V, mprime)
    U[:,1:mprime], λ[1:mprime], V[:,1:mprime]
end

summary(svd_trunc::TruncBond) = "SVD truncation to bond size m'="*string(svd_trunc.mprime)

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

function summary(svd_trunc::TruncBondMax)
    "SVD truncation to bond size m'="*string(svd_trunc.mprime)*
        ". Max error "*string(only(svd_trunc.maxerr))
end

summary_compact(svd_trunc::SVDTrunc) = summary(svd_trunc)

function summary_compact(svd_trunc::Union{TruncBond,TruncBondMax}) 
    ("SVD Matrix size", string(svd_trunc.mprime))
end

function summary_compact(svd_trunc::TruncThresh) 
    ("SVD tolerance", string(svd_trunc.ε))
end

struct TruncBondThresh{T} <: SVDTrunc
    mprime :: Int
    ε :: T
    TruncBondThresh(mprime::Int, ε::T=0.0) where T = new{T}(mprime, ε)
end
function (svd_trunc::TruncBondThresh)(M::AbstractMatrix)
    U, λ, V = svd(M)
    λ_norm = norm(λ)
    mprime = min(
        findlast(λₖ > svd_trunc.ε*λ_norm for λₖ in λ),
        length(λ), 
        svd_trunc.mprime
        )
    _debug_svd(M, U, λ, V, mprime)
    U[:,1:mprime], λ[1:mprime], V[:,1:mprime]
end

summary(svd_trunc::TruncBondThresh) = "SVD truncation with truncation to bond size given by the minimum of threshold ε="*string(svd_trunc.ε)*
    " and m'="*string(svd_trunc.mprime)

function summary_compact(svd_trunc::TruncBondThresh) 
    ("SVD tolerance, m'", string(svd_trunc.ε)*", "*string(svd_trunc.mprime))
end