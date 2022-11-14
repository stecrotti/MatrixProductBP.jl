function exact_prob(bp::MPBP{q,T,F,U}) where {q,T,F,U}
    @assert q==2 "Can compute exact prob only for binary variables (for now)"
    @unpack g, w, p⁰, ϕ, ψ = bp
    N = nv(g)
    T*N > 15 && @warn "T*N=$(T*N). This will take some time!"
    
    p = ones(2^(N*(T+1)))
    prog = Progress(2^(N*(T+1)), desc="Computing joint probability")
    X = zeros(Int, T+1, N)
    for x in 1:2^(N*(T+1))
        X .= int_to_matrix(x-1, (T+1,N))
        for i in 1:N
            p[x] *= p⁰[i][X[1,i]]
            ∂i = neighbors(g, i)
            for t in 1:T
                p[x] *= w[i][t](X[t+1,i], X[t,∂i], X[t,i])
                p[x] *= ϕ[i][t][X[t+1,i]]
            end
        end
        for (i, j, ij) in edges(g)
            for t in 1:T
                p[x] *= sqrt( ψ[ij][t][X[t+1,i],X[t+1,j]] )
            end
        end
        next!(prog)
    end
    Z = sum(p)
    p ./= Z
    p, Z
end