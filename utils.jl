# maps (1,2) -> (1,-1)
potts2spin(x::Integer) = 3-2x

struct AllOneTensor; end
Base.getindex(::AllOneTensor, idx...) = 1
Base.size(::AllOneTensor, i::Integer) = 1
Base.axes(::AllOneTensor, i::Integer) = 1:1

# symmetrize and set diagonal to zero
function symmetrize_nodiag!(A::AbstractMatrix)
    A .= (A+A')/2
    for i in axes(A, 1)
        A[i,i] = 0
    end
    A
end
