using BEAST
using ExaFMMt
using LinearAlgebra
using LinearMaps
using SparseArrays


struct FMMMatrixSL{I, F <: Real, K} <: LinearMaps.LinearMap{K}
    fmm::ExaFMMt.ExaFMM{K}
    B_trial::SparseMatrixCSC{F, I}
    B_test::SparseMatrixCSC{F, I}
    BtCB::HMatrix{I, K}
    fullmat::HMatrix{I, K}
    rowdim::I
    columndim::I
end

function Base.size(fmat::FMMMatrixSL, dim=nothing)
    if dim === nothing
        return (fmat.rowdim, fmat.columndim)
    elseif dim == 1
        return fmat.rowdim
    elseif dim == 2
        return fmat.columndim
    else
        error("dim must be either 1 or 2")
    end
end

function Base.size(fmat::Adjoint{T}, dim=nothing) where T <: FMMMatrixSL
    if dim === nothing
        return reverse(size(adjoint(fmat)))
    elseif dim == 1
        return size(adjoint(fmat),2)
    elseif dim == 2
        return size(adjoint(fmat),1)
    else
        error("dim must be either 1 or 2")
    end
end

@views function LinearAlgebra.mul!(y::AbstractVecOrMat, A::FMMMatrixSL, x::AbstractVector)
    LinearMaps.check_dim_mul(y, A, x)

    if eltype(x) != eltype(A)
        x = eltype(A).(x)
    end
    fill!(y, zero(eltype(y)))

    y .= A.B_test * conj.(A.fmm * conj.(A.B_trial * x))[:,1] - A.BtCB * x + A.fullmat * x

    return y
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    A::LinearMaps.TransposeMap{<:Any,<:FMMMatrixSL},
    x::AbstractVector
)
    LinearMaps.check_dim_mul(y, A, x)

    if eltype(x) != eltype(A)
        x = eltype(A).(x)
    end
    fill!(y, zero(eltype(y)))

    y .= A.B_test * conj.(A.fmm * conj.(A.B_trial * x))[:,1] - A.BtCB * x + A.fullmat * x

    return y
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    A::LinearMaps.AdjointMap{<:Any,<:FMMMatrixSL},
    x::AbstractVector
)
    LinearMaps.check_dim_mul(y, A, x)

    if eltype(x) != eltype(A)
        x = eltype(A).(x)
    end

    fill!(y, zero(eltype(y)))

    y .= A.B_test * conj.(A.fmm * conj.(A.B_trial * x))[:,1] - A.BtCB * x + A.fullmat * x

    return y
end

function FMMMatrix(
    op::BEAST.HH3DSingleLayerFDBIO,
    test_functions::BEAST.Space, 
    trial_functions::BEAST.Space, 
    testqp::Matrix,
    trialqp::Matrix,
    fmm::ExaFMMt.ExaFMM{K},
    BtCB::HMatrix{I, K},
    fullmat::HMatrix{I, K},
) where {I, K}

    B, B_test = sample_basisfunctions(op, test_functions, trial_functions, testqp, trialqp)
    
    return FMMMatrixSL(
        fmm,
        B,
        B_test,
        BtCB,
        fullmat,
        size(fullmat)[1],
        size(fullmat)[2]
    )

end

function sample_basisfunctions(
    op::BEAST.HH3DSingleLayerFDBIO,
    test_functions::BEAST.Space, 
    trial_functions::BEAST.Space, 
    testqp::Matrix,
    trialqp::Matrix
)   
    rc, vals = sample_basisfunctions(op, trialqp, trial_functions)
    B = dropzeros(sparse(rc[:, 1], rc[:, 2], vals)) 
    B_test = B

    if test_functions != trial_functions 
        rc_test, vals_test = sample_basisfunctions(op, testqp, test_functions)
        B_test = dropzeros(sparse(rc_test[:, 1], rc_test[:, 2], vals_test))
    else
        B_test = sparse(transpose(B))
    end

    return B, B_test
end
