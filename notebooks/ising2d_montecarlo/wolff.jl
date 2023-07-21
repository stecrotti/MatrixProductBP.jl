using LinearAlgebra, SparseArrays, ProgressMeter

chain(n, open::Bool=true)=(x=spdiagm(-1=>fill(true,n-1),1=>fill(true,n-1)); if !open x[n,1]=x[1,n]=true; end; x)
cartesian(A,B) = kron(A,I(size(B,1))) .| kron(I(size(A,1)),B)
lattice(L, bc=trues(length(L))) = mapreduce(chain, cartesian, reverse(L), reverse(bc), init=spzeros(Bool,1,1))

function wolff(βJ::SparseMatrixCSC, βh::AbstractVector, nsteps, 
        statistics = (t, σ)->nothing; ntherm::Integer=0)
    βJh = [ spzeros(1,1)  βh';
            βh            βJ ]
    @assert length(βh) == size(βJ, 1)
    return wolff(βJh, nsteps, (t, σ) -> statistics(t, σ[2:end]*σ[1]); ntherm)
end

function wolff(βJ::SparseMatrixCSC, nsteps, 
        statistics = (t, σ)->nothing; ntherm::Integer=0)
    n = size(βJ, 1)
    @assert issymmetric(βJ)
    σ = rand((-1,1), n)
    R = fill(false, n)
    Q = Int[]
    rows, vals = rowvals(βJ), nonzeros(βJ)
    function growregion()
        R[Q] .= false
        empty!(Q)
        push!(Q, rand(1:n))
        R[Q[1]] = true
        ∂(i) = @views zip(rows[nzrange(βJ, i)], vals[nzrange(βJ, i)])
        k = 1
        while k <= length(Q)
            i = Q[k]
            for (j,βJij) in ∂(i)
                v = σ[i]*σ[j]*βJij
                if v > 0 && R[j] == false && rand() > exp(-2v)
                    R[j] = true
                    push!(Q, j)
                end
            end
            k += 1
        end
    end
    for t=1:ntherm
        growregion()
        σ[Q] .*= -1
    end
    @showprogress for t=1:nsteps
        growregion()
        σ[Q] .*= -1
        statistics(t, σ)
    end
end

mutable struct Stats
    s :: Vector{Float64}
    m :: Vector{Float64}
    n :: Int64
    Stats(N::Integer) = new(zeros(N), zeros(0), 0) 
end

magnetizations(stats::Stats) = stats.s ./ stats.n
magnetization(stats::Stats) = stats.m

function (stats::Stats)(t, σ) 
    stats.s .+= σ
    push!(stats.m, mean(σ))
    stats.n += 1
    nothing
end