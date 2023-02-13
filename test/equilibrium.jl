# Test that population code with concentrated degree distributions gives the same results
#  as code with fixed degree

J = 0.8
h = 0.0
β = 1.0
k = 3

m_distr = equilibrium_magnetization(RandomRegular(k); pJ=Dirac(J), ph=Dirac(h), β, popsize=10^5, 
    tol=1e-15, maxiter=100)
m_fp = equilibrium_magnetization(RandomRegular(k), J; β, h, tol=1e-15)
m_distr, m_fp
@test m_distr ≈ m_fp