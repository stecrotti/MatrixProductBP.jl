# Other methods for comparisons

function initialize(T, Δt, g, λ_rate, ρ_rate, γ)
    Tdisc = floor(Int, T/Δt)
    ϕ = [zeros(Tdisc+1) for _ in edges(g)]
    r = [fill(γ[i], Tdisc+1) for i in vertices(g)]
    for (i,j,id) in edges(g)
        ϕ[id][1] = r[i][1]*(1-r[j][1])
    end
    λ = λ_rate * Δt
    ρ = ρ_rate * Δt
    prog = Progress(Tdisc)
    return Tdisc, λ, ρ, prog, r, ϕ
end

### Discrete time

# dynamic message passing
function dmp_disc(T, Δt, g, λ_rate, ρ_rate, γ)
    Tdisc, λ, ρ, prog, r, ϕ = initialize(T, Δt, g, λ_rate, ρ_rate, γ)
    for t in 2:Tdisc+1
        for i in vertices(g)
           r[i][t] = (1-ρ)*r[i][t-1] + 
                (1-r[i][t-1])*(1-prod(1-λ*ϕ[idx(e)][t-1] for e in inedges(g, i); init=1.0))
        end
        for (i, j, id) in edges(g)
           ϕ[id][t] = (1-ρ)*ϕ[id][t-1] + 
                (1-ϕ[id][t-1])*(1-prod(1-λ*ϕ[idx(e)][t-1] for e in inedges(g, i) if src(e)!=j; init=1.0))
        end
        next!(prog)
    end
    r, ϕ
end

# individual-based mean field
function ibmf_disc(T, Δt, g, λ_rate, ρ_rate, γ)
    Tdisc, λ, ρ, prog, r = initialize(T, Δt, g, λ_rate, ρ_rate, γ)
    for t in 2:Tdisc+1
        for i in vertices(g)
            r[i][t] = (1-ρ)*r[i][t-1] +  
                (1-r[i][t-1])*(1-prod(1-λ*r[src(e)][t-1] for e in inedges(g, i); init=1.0))
        end
        next!(prog)
    end
    r
end


### Continuous time

# cavity master equation
function cme!(T, ΔT, g, λ_rate, ρ_rate, γ)
    λ, ρ, prog, r, ϕ = initialize(T, Δt, g, λ_rate, ρ_rate, γ)
    for t in 2:Tdisc+1
        for i in vertices(g)
           r[t][i] = (1-ρ)*r[t-1][i] + λ*(1-r[t-1][i])*sum(ϕ[t-1][idx(e)] for e in inedges(g, i); init=0.0)
        end
        for (i, j, id) in edges(g)
           ϕ[t][id] = (1-ρ)*ϕ[t-1][id] + 
                (1-ϕ[t-1][id])*λ*sum(ϕ[t-1][idx(e)] for e in inedges(g, i) if src(e)!=j; init=0.0)
        end
    end
    nothing
end