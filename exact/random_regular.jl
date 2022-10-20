using Roots

function F(β, J, k::Integer)
    f(m) = tanh((atanh(m))/k) - m*tanh(β*J)
end

function rs_magnetization_fixedpoints(β, J, k::Integer)
    f = F(β, J, k)
    find_zeros(f, (-1,1))
end