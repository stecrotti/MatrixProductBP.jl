import Base.ComplexF64
function complex(x::AcbFieldElem)
    return ComplexF64(Float64(real(x)), Float64(imag(x)))
end

#=
    _fourier3int(α::Int, γ::Int; w::Float64=0.5, J::Float64=1.0, P::Float64=2.0)

Computes the integral ``∫dx u_α(x) u*_γ(Jx)``. The integrations ranges are ``[-w,+w]``. This is part of the calculation of the integral ``∫dx₁ ∫dx₂ u_α(x₁) u_β(x₂) u*_γ(J₁x₁+J₂x₂)``.
=#
function _fourier3int(α::Int, γ::Int, w::Real, J::Float64, Pscale::Float64, d₁::Integer, d₂::Integer)
    d₁₂ = d₁ + d₂
    kαγ = 2π * (d₁₂*α - d₁*J*γ) / (d₁*d₁₂*Pscale)
    iszero(kαγ) ? w : sin(kαγ*w)/kαγ
end

function compute_prob_ys(wᵢ::Vector{U}, qi::Int, μin::Vector{M2}, ψout, T, svd_trunc) where {U<:FourierBPFactor, M2<:AbstractMPEM2}
    dᵢ = length(μin)
    wᵢᵗ = wᵢ[1]
    K, σ, P = wᵢᵗ.K, wᵢᵗ.σ, wᵢᵗ.P
    J, β = float.(wᵢᵗ.J), wᵢᵗ.β
    scale = wᵢᵗ.scale

    μ_fourier = [fourier_tensor_train_spin(μ, K, P*scale, σ) for μ in μin]

    function op((F₁, J₁, d₁), (F₂, J₂, d₂))
        K1 = (size(F₁[1],3)-1)/2 |> Int
        K2 = (size(F₂[1],3)-1)/2 |> Int
        @tullio avx=false Int_1[γ,α] :=  _fourier3int(α,γ,d₁*scale,J₁,P*scale,d₁,d₂) α∈-K1:K1, γ∈-K:K
        @tullio avx=false Int_2[γ,β] :=  _fourier3int(β,γ,d₂*scale,J₂,P*scale,d₂,d₁) β∈-K2:K2, γ∈-K:K
    
        GG = map(zip(F₁,F₂)) do (F₁ᵗ, F₂ᵗ)
            @tullio Gt1[m1,n1,γ,x] := F₁ᵗ[m1,n1,α,x] * Int_1[γ,α]
            @tullio Gt2[m2,n2,γ,x] := F₂ᵗ[m2,n2,β,x] * Int_2[γ,β]
            @tullio Gt[m1,m2,n1,n2,γ,x] := 4/(P*scale*(d₁+d₂)) * Gt1[m1,n1,γ,x] * Gt2[m2,n2,γ,x]
            @cast Gᵗ[(m1,m2),(n1,n2),γ,x] := Gt[m1,m2,n1,n2,γ,x]
            return collect(Gᵗ)
        end
    
        G = fourier_tensor_train(GG, z=F₁.z*F₂.z)
        compress!(G; svd_trunc)
        normalize_eachmatrix!(G)
        any(any(isnan, Gᵗ) for Gᵗ in G) && @error "NaN in Fourier tensor train"
        return (G, 1.0, d₁+d₂)
    end

    ginit = [[(1.0 + 0.0*im)/(P*scale) for _ in 1:1, _ in 1:1, α in -K:K, _ in 1:2] for _ in 1:T+1] |> fourier_tensor_train
    Ginit = (ginit, 1.0, 1)

    dest, (full,) = cavity(zip(μ_fourier, J, fill(1,length(μ_fourier))) |> collect, op, Ginit)
    (C,) = unzip(dest)
    return C, full, μ_fourier
end
compute_prob_ys(wᵢ::Vector{U}, qi::Int, μin::Vector{M2}, ψout, T, svd_trunc) where {U<:FourierBPFactor, M2<:InfiniteUniformMPEM2} = throw(ArgumentError("Not implemented"))

function _integral_hypgeom(β,Jⱼᵢ,xⱼᵗ,hᵢ,xᵢᵗ⁺¹,kᵧ, yy)
    function _primitive_1(X,β,Jxj,bb)
        hypgeom = 0.0 + 0.0im
        precbits = 64
        while true
            CC = AcbField(precbits)
            a_ = CC(1.0)
            b_ = CC(bb)
            c_ = CC(1+bb)
            x_ = CC(-exp(2β*(Jxj+X)))
            hyp_nemo = hypergeometric_2f1(a_, b_, c_, x_)
            
            if !isnan(Float64(real(hyp_nemo))+Float64(real(hyp_nemo)))
                max((Nemo.radius(real(hyp_nemo))), Nemo.radius(imag(hyp_nemo))) > 1e-12 && @error "Possible numerical instability in hypergeometric function ($hyp_nemo)"
                hypgeom = complex(hyp_nemo)
                break
            end
            precbits *= 2
        end

        isnan(hypgeom) && error("NaN in hypergeometric function")
        
        return hypgeom
    end
    function _primitive_2(X)
        a = β * (1 + exp(2β * (Jxj+X)))
        return Jxj + X - log(a) / (2β)
    end

    Jxj = Jⱼᵢ*xⱼᵗ + hᵢ
    if iszero(kᵧ) && xᵢᵗ⁺¹==-1
        return _primitive_2(yy) - _primitive_2(-yy)
    else
        xp1 = 1+xᵢᵗ⁺¹
        bb = (xp1 + im*kᵧ/β) / 2
        denom = bb * 2 * β

        exp_prim_p1 = exp(im*kᵧ*yy + β*xp1*(Jxj+yy)) * _primitive_1(yy,β,Jxj,bb)
        exp_prim_m1 = exp(-im*kᵧ*yy + β*xp1*(Jxj-yy)) * _primitive_1(-yy,β,Jxj,bb)
        return (exp_prim_p1 - exp_prim_m1) / denom
    end
end


function _f_bp_partial(A::MPEM2, wᵢ::Vector{U}, ϕᵢ, 
        d::Integer, prob::Function, qj, j) where {U<:FourierBPFactor}
    wᵢᵗ = wᵢ[1]
    K, P = wᵢᵗ.K, wᵢᵗ.P
    J, hᵢ, β = float(wᵢᵗ.J[j]), wᵢᵗ.h, wᵢᵗ.β
    p = wᵢᵗ.p
    scale = wᵢᵗ.scale
    Pscale = d * scale * P
    ybound = d * P / 2

    @tullio avx=false In[γ,xᵢᵗ⁺¹,xⱼᵗ] := _integral_hypgeom(β,J,potts2spin(xⱼᵗ),hᵢ,potts2spin(xᵢᵗ⁺¹),2π*γ/Pscale, ybound) γ∈-K:K, xᵢᵗ⁺¹∈1:2, xⱼᵗ∈1:qj
    any(isnan, In) && error("NaN in integral")

    B = map(eachindex(A)) do t
        Aᵗ, ϕᵢᵗ = A[t], ϕᵢ[t]
        @tullio C[m,n,xᵢᵗ,xⱼᵗ,xᵢᵗ⁺¹] := Aᵗ[m,n,γ,xᵢᵗ] * In[γ,xᵢᵗ⁺¹,xⱼᵗ]
        @tullio Bᵗ[m,n,xᵢᵗ,xⱼᵗ,xᵢᵗ⁺¹] := (p * (Pscale) * (xᵢᵗ⁺¹==xᵢᵗ) * Aᵗ[m,n,0,xᵢᵗ] + (1-p) * C[m,n,xᵢᵗ,xⱼᵗ,xᵢᵗ⁺¹]) * ϕᵢᵗ[xᵢᵗ]
        return Bᵗ
    end
    Aᵀ, ϕᵢᵀ = A[end], ϕᵢ[end]
    @tullio Bᵀ[m,n,xᵢᵗ,xⱼᵗ,xᵢᵗ⁺¹] := Aᵀ[m,n,0,xᵢᵗ] * (Pscale) * ϕᵢᵀ[xᵢᵗ] xⱼᵗ∈1:qj, xᵢᵗ⁺¹∈1:2
    B[end] = Bᵀ

    return MPEM3(collect.(B), z=A.z)
end

# compute matrix B for m{∂i∖j→i}
function f_bp_partial_ij(A::AbstractMPEM2, wᵢ::Vector{U}, ϕᵢ, d::Integer, qj, j) where {U<:FourierBPFactor}
    _f_bp_partial(A, wᵢ, ϕᵢ, d+1, x->x, qj, j)
end

# compute matrix B for bᵢ
function f_bp_partial_i(A::AbstractMPEM2, wᵢ::Vector{U}, ϕᵢ, d::Integer) where {U<:FourierBPFactor}
    w = FourierGlauberFactor([0.0], wᵢ[1].h, wᵢ[1].β, wᵢ[1].K, wᵢ[1].σ, wᵢ[1].P, wᵢ[1].scale, wᵢ[1].p)
    _f_bp_partial(A, fill(w,length(A)), ϕᵢ, d+1, x->x, 1, 1)
end