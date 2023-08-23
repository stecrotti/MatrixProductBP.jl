using QuadGK

function onsager_energy(J, β)
    function integral(J, β)
        k = 1 / (sinh(2β*J)^2) 
        f(θ) = 1 / sqrt(1 - 4/(1/k+2+k)*sin(θ)^2)
        quadgk(f, 0, π/2, rtol=1e-10)
    end
    int = integral(J, β)[1]
    -J*coth(2β*J)*(1 + 2/π*(2tanh(2β*J)^2-1)*int)
end