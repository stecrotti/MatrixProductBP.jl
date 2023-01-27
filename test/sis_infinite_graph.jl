

T = 6         # final time
k = 3          # degree
γ = 0.1       # prob. of zero patient
ΔT = 1.0      # discretization
λ = 0.1        # rate of transmission
ρ = 0.2        # rate of recovery

wᵢ = fill(Models.SISFactor(λ, ρ), floor(Int, T/ΔT) + 1)
ϕᵢ = [ t == 0 ? [1-γ, γ] : ones(2) for t in 0:floor(Int, T/ΔT)]
bp = mpbp_infinite_graph(k, wᵢ, 2, ϕᵢ)

svd_trunc = TruncBond(10)
maxiter = 200
tol = 1e-14
iters = iterate!(bp; maxiter, svd_trunc, tol);

@testset "SIS infinite graph" begin

@test beliefs(bp)[1] ≈ [
    [0.9000000001671186, 0.0999999998328814],
    [0.8932690998131098, 0.10673090018689023],
    [0.8899420329322244, 0.11005796706777556],
    [0.8884643888492034, 0.11153561115079656],
    [0.8880305235706524, 0.1119694764293476],
    [0.8882121515614524, 0.11178784843854758],
    [0.8887717202217936, 0.1112282797782064]]
end