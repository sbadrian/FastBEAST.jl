using BEAST
using ExaFMMt
using LinearAlgebra
using LinearMaps
using SparseArrays

struct FMMMatrixMWSL{I, F <: Real, K} <: LinearMaps.LinearMap{K}
    fmm::ExaFMMt.ExaFMM{K}
    fmm_t::ExaFMMt.ExaFMM{K}
    op::BEAST.MWSingleLayer3D
    B1::SparseMatrixCSC{F, I}
    B2::SparseMatrixCSC{F, I}
    B3::SparseMatrixCSC{F, I}
    Bdiv::SparseMatrixCSC{F, I}
    B1_test::SparseMatrixCSC{F, I}
    B2_test::SparseMatrixCSC{F, I}
    B3_test::SparseMatrixCSC{F, I}
    Bdiv_test::SparseMatrixCSC{F, I}
    BtCB::HMatrix{I, K}
    fullmat::HMatrix{I, K}
    rowdim::I
    columndim::I
end

function Base.size(fmat::FMMMatrixMWSL, dim=nothing)
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

function Base.size(fmat::Adjoint{T}, dim=nothing) where T <: FMMMatrixMWSL
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

@views function LinearAlgebra.mul!(y::AbstractVecOrMat, A::FMMMatrixMWSL, x::AbstractVector)
    LinearMaps.check_dim_mul(y, A, x)

    if eltype(x) != eltype(A)
        x = eltype(A).(x)
    end
    fill!(y, zero(eltype(y)))

    res1 = A.B1_test * (A.fmm * (A.B1 * x))[:,1]
    res2 = A.B2_test * (A.fmm * (A.B2 * x))[:,1]
    res3 = A.B3_test * (A.fmm * (A.B3 * x))[:,1]

    y1 = (A.op.α .* (res1 + res2 + res3))

    y2 = - (A.op.β) .*
        (A.Bdiv_test * (A.fmm * (A.Bdiv * x))[:,1])

    y.= (y1 - y2) - A.BtCB * x + A.fullmat * x
    
    return y
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    A::LinearMaps.TransposeMap{<:Any,<:FMMMatrixMWSL},
    x::AbstractVector
)
    LinearMaps.check_dim_mul(y, A, x)

    if eltype(x) != eltype(A)
        x = eltype(A).(x)
    end
    fill!(y, zero(eltype(y)))

    res1 = A.B1_test * (A.fmm * (A.B1 * x))[:,1]
    res2 = A.B2_test * (A.fmm * (A.B2 * x))[:,1]
    res3 = A.B3_test * (A.fmm * (A.B3 * x))[:,1]

    y1 = (A.op.α .* (res1 + res2 + res3))

    y2 = - (A.op.β) .*
        (A.Bdiv_test * (A.fmm * (A.Bdiv * x))[:,1])

    y.= (y1 - y2) - A.BtCB * x + A.fullmat * x

    return y
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    At::LinearMaps.AdjointMap{<:Any,<:FMMMatrixMWSL},
    x::AbstractVector
)

    mul!(y, transpose(adjoint(At)), conj(x))

    return conj!(y)
end

function FMMMatrix(
    op::BEAST.MWSingleLayer3D,
    test_functions::BEAST.Space, 
    trial_functions::BEAST.Space, 
    testqp::Matrix,
    trialqp::Matrix,
    fmm::ExaFMMt.ExaFMM{K},
    fmm_t::ExaFMMt.ExaFMM{K},
    BtCB::HMatrix{I, K},
    fullmat::HMatrix{I, K},
) where {I, K}

    B1, B2, B3, Bdiv, B1_test, B2_test, B3_test, Bdiv_test = sample_basisfunctions(
        op,
        test_functions, 
        trial_functions, 
        testqp,
        trialqp,
    )    

    return FMMMatrixMWSL(
        fmm,
        fmm_t,
        op,
        B1,
        B2,
        B3,
        Bdiv,
        B1_test,
        B2_test,
        B3_test,
        Bdiv_test,
        BtCB,
        fullmat,
        size(fullmat)[1],
        size(fullmat)[2]
    )
end

function sample_basisfunctions(
    op::BEAST.MWSingleLayer3D, 
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

    rcdiv, valsdiv = sample_divbasisfunctions(trialqp, trial_functions)
    Bdiv = dropzeros(sparse(rcdiv[:, 1], rcdiv[:, 2], valsdiv))
    Bdiv_test = Bdiv

    if test_functions != trial_functions
        rc_test,  vals_test = sample_basisfunctions(op, testqp, test_functions)
        B1_test = dropzeros(sparse(rc_test[:, 2], rc_test[:, 1], vals_test[:, 1]))
        B2_test = dropzeros(sparse(rc_test[:, 2], rc_test[:, 1], vals_test[:, 2]))
        B3_test = dropzeros(sparse(rc_test[:, 2], rc_test[:, 1], vals_test[:, 3]))
        rcdiv_test, valsdiv_test = sample_divbasisfunctions(testqp, test_functions)
        Bdiv_test = dropzeros(sparse(rcdiv_test[:, 2], rcdiv_test[:, 1], valsdiv_test))
    else
        B1_test = sparse(transpose(B1_test))
        B2_test = sparse(transpose(B2_test))
        B3_test = sparse(transpose(B3_test)) 
        Bdiv_test = sparse(transpose(Bdiv_test))
    end

    return B1, B2, B3, Bdiv, B1_test, B2_test, B3_test, Bdiv_test
end