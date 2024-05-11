using BEAST
using ExaFMMt
using LinearAlgebra
using LinearMaps
using SparseArrays

struct FMMMatrixMWDL{I, F <: Real, K} <: LinearMaps.LinearMap{K}
    fmm::ExaFMMt.ExaFMM{K}
    fmm_t::ExaFMMt.ExaFMM{K}
    op::BEAST.MWDoubleLayer3D
    B1::SparseMatrixCSC{F, I}
    B2::SparseMatrixCSC{F, I}
    B3::SparseMatrixCSC{F, I}
    B1_test::SparseMatrixCSC{F, I}
    B2_test::SparseMatrixCSC{F, I}
    B3_test::SparseMatrixCSC{F, I}
    BtCB::HMatrix{I, K}
    fullmat::HMatrix{I, K}
    rowdim::I
    columndim::I
end

function Base.size(fmat::FMMMatrixMWDL, dim=nothing)
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

function Base.size(fmat::Adjoint{T}, dim=nothing) where T <: FMMMatrixMWDL
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

@views function LinearAlgebra.mul!(y::AbstractVecOrMat, A::FMMMatrixMWDL, x::AbstractVector)
    LinearMaps.check_dim_mul(y, A, x)

    if eltype(x) != eltype(A)
        x = eltype(A).(x)
    end
    fill!(y, zero(eltype(y)))

    res1 = (A.fmm * (A.B1 * x))[:,2:4]
    res2 = (A.fmm * (A.B2 * x))[:,2:4]
    res3 = (A.fmm * (A.B3 * x))[:,2:4]

    y1 = A.B1_test * (res3[:,2] - res2[:,3])
    y2 = A.B2_test * (res1[:,3] - res3[:,1])
    y3 = A.B3_test * (res2[:,1] - res1[:,2])

    y.= (y1 + y2 + y3) - A.BtCB * x + A.fullmat * x

    return y
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    A::LinearMaps.TransposeMap{<:Any,<:FMMMatrixMWDL},
    x::AbstractVector
)
    LinearMaps.check_dim_mul(y, A, x)

    if eltype(x) != eltype(A)
        x = eltype(A).(x)
    end
    fill!(y, zero(eltype(y)))

    res1 = (A.fmm * (A.B1 * x))[:,2:4]
    res2 = (A.fmm * (A.B2 * x))[:,2:4]
    res3 = (A.fmm * (A.B3 * x))[:,2:4]

    y1 = A.B1_test * (res3[:,2] - res2[:,3])
    y2 = A.B2_test * (res1[:,3] - res3[:,1])
    y3 = A.B3_test * (res2[:,1] - res1[:,2])

    y.= (y1 + y2 + y3) - A.BtCB * x + A.fullmat * x

    return A.op.alpha .* y
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    A::LinearMaps.AdjointMap{<:Any,<:FMMMatrixMWDL},
    x::AbstractVector
)
    LinearMaps.check_dim_mul(y, A, x)

    if eltype(x) != eltype(A)
        x = eltype(A).(x)
    end
    fill!(y, zero(eltype(y)))

    res1 = (A.fmm * (A.B1 * x))[:,2:4]
    res2 = (A.fmm * (A.B2 * x))[:,2:4]
    res3 = (A.fmm * (A.B3 * x))[:,2:4]

    y1 = A.B1_test * (res3[:,2] - res2[:,3])
    y2 = A.B2_test * (res1[:,3] - res3[:,1])
    y3 = A.B3_test * (res2[:,1] - res1[:,2])

    y.= A.op.alpha .* (y1 + y2 + y3) - A.BtCB * x + A.fullmat * x

    return y
end

function FMMMatrix(
    op::BEAST.MWDoubleLayer3D,
    test_functions::BEAST.Space, 
    trial_functions::BEAST.Space, 
    testqp::Matrix,
    trialqp::Matrix,
    fmm::ExaFMMt.ExaFMM{K},
    fmm_t::ExaFMMt.ExaFMM{K},
    BtCB::HMatrix{I, K},
    fullmat::HMatrix{I, K},
) where {I, K}

    B1, B2, B3, B1_test, B2_test, B3_test = sample_basisfunctions(
        op,
        test_functions,
        trial_functions,
        testqp,
        trialqp
    )
      
    return FMMMatrixMWDL(
        fmm,
        fmm_t,
        op,
        B1,
        B2,
        B3,
        B1_test,
        B2_test,
        B3_test,
        BtCB,
        fullmat,
        size(fullmat)[1],
        size(fullmat)[2]
    )

end

function sample_basisfunctions(
    op::BEAST.MWDoubleLayer3D,
    test_functions::BEAST.Space, 
    trial_functions::BEAST.Space, 
    testqp::Matrix,
    trialqp::Matrix,
)

    rc, vals = sample_basisfunctions(op, trialqp, trial_functions)
    B1 = dropzeros(sparse(rc[:, 1], rc[:, 2], vals[:, 1]))
    B2 = dropzeros(sparse(rc[:, 1], rc[:, 2], vals[:, 2]))
    B3 = dropzeros(sparse(rc[:, 1], rc[:, 2], vals[:, 3]))
    B1_test, B2_test, B3_test = B1, B2, B3

    if test_functions != trial_functions
        rc_test,  vals_test = sample_basisfunctions(op, testqp, test_functions)
        B1_test = dropzeros(sparse(rc_test[:, 2], rc_test[:, 1], vals_test[:, 1]))
        B2_test = dropzeros(sparse(rc_test[:, 2], rc_test[:, 1], vals_test[:, 2]))
        B3_test = dropzeros(sparse(rc_test[:, 2], rc_test[:, 1], vals_test[:, 3]))
    else
        B1_test = sparse(transpose(B1_test))
        B2_test = sparse(transpose(B2_test))
        B3_test = sparse(transpose(B3_test)) 
    end

    return B1, B2, B3, B1_test, B2_test, B3_test
end
