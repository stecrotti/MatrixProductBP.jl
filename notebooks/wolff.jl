using LinearAlgebra, SparseArrays, ProgressMeter

chain(n, open::Bool=true)=(x=spdiagm(-1=>fill(true,n-1),1=>fill(true,n-1)); if !open x[n,1]=x[1,n]=true; end; x)
cartesian(A,B) = kron(A,I(size(B,1))) .| kron(I(size(A,1)),B)
lattice(L, bc=trues(length(L))) = mapreduce(chain, cartesian, reverse(L), reverse(bc), init=spzeros(Bool,1,1))

function wolff(βJ::SparseMatrixCSC, βh::AbstractVector, nsteps, 
        statistics = (t, σ)->nothing; ntherm::Integer=0, showprogress=false)
    # βJh = [ spzeros(1,1)  βh';
    #         βh            βJ ]
    βJh = [ βJ  βh;
            βh'  spzeros(1,1) ]
    @assert length(βh) == size(βJ, 1)
    return wolff(βJh, nsteps, (t, σ, βJ) -> statistics(t, σ[1:end-1]*σ[end], βJ); 
        ntherm, showprogress)
end

function wolff(βJ::SparseMatrixCSC, nsteps, 
        statistics = (t, σ, βJ)->nothing; ntherm::Integer=0, showprogress=true)
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
    prog = Progress(nsteps, desc = "Running MH with Wolff proposals", dt = showprogress ? 0.1 : Inf)
    for t=1:nsteps
        growregion()
        σ[Q] .*= -1
        statistics(t, σ, βJ)
        next!(prog)
    end
end

mutable struct Stats
    s :: Vector{Float64}
    m :: Vector{Float64}
    n :: Int64
    c :: Matrix{Float64}
    Stats(N::Integer) = new(zeros(N), zeros(0), 0, zeros(N,N)) 
end

magnetizations(stats::Stats) = stats.s ./ stats.n
magnetization(stats::Stats) = stats.m
correlations(stats::Stats) = stats.c ./ stats.n

function _correlations!(c, σ, βJ)
    rows = rowvals(βJ)
    for j in axes(βJ, 2)
        for k in nzrange(βJ, j)
            i = rows[k]
            (j > length(σ) || i > length(σ)) && continue
            c[i,j] += σ[i] * σ[j]
        end
    end
end

function (stats::Stats)(t, σ, βJ) 
    stats.s .+= σ
    _correlations!(stats.c, σ, βJ)
    push!(stats.m, mean(σ))
    stats.n += 1
    nothing
end