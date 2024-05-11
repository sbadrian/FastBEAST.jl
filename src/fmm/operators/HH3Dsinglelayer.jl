using BEAST
using ExaFMMt
using LinearAlgebra
using LinearMaps
using SparseArrays


struct FMMMatrixSL{I, F <: Real, K, KE} <: LinearMaps.LinearMap{K}
    fmm::ExaFMMt.ExaFMM{KE}
    fmm_t::ExaFMMt.ExaFMM{KE}
    op::BEAST.HH3DSingleLayerFDBIO
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

    fill!(y, zero(eltype(y)))

    if eltype(x) <: Complex && eltype(A.fmm) <: Real
        y .+= mul!(copy(y), A, real.(x))
        y .+= im .* mul!(copy(y), A, imag.(x)) 
        return y
    end

    if eltype(x) != eltype(A.fmm)
        xfmm = eltype(A.fmm).(x)
    else
        xfmm = x
    end

    y .= A.op.alpha .* (A.B_test * (A.fmm * (A.B_trial * xfmm))[:,1]) - A.BtCB * x + A.fullmat * x

    return y
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    At::LinearMaps.TransposeMap{<:Any,<:FMMMatrixSL},
    x::AbstractVector
)
    LinearMaps.check_dim_mul(y, At, x)

    fill!(y, zero(eltype(y)))

    A = At.lmap

    if eltype(x) <: Complex && eltype(A.fmm) <: Real
        y .+= mul!(copy(y), At, real.(x))
        y .+= im .* mul!(copy(y), At, imag.(x))
        return y
    end

    if eltype(x) != eltype(A.fmm)
        xfmm = eltype(A.fmm).(x)
    else
        xfmm = x
    end

    y .= A.op.alpha .* (transpose(A.B_trial) * (A.fmm_t * (transpose(A.B_test) *  xfmm))[:,1]) - transpose(A.BtCB) * x + transpose(A.fullmat) * x

    return y
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    At::LinearMaps.AdjointMap{<:Any,<:FMMMatrixSL},
    x::AbstractVector
)

    mul!(y, transpose(adjoint(At)), conj(x))

    return conj!(y)
end

function FMMMatrix(
    op::BEAST.HH3DSingleLayerFDBIO,
    test_functions::BEAST.Space, 
    trial_functions::BEAST.Space, 
    testqp::Matrix,
    trialqp::Matrix,
    fmm::ExaFMMt.ExaFMM{KE},
    fmm_t::ExaFMMt.ExaFMM{KE},
    BtCB::HMatrix{I, K},
    fullmat::HMatrix{I, K},
) where {I, K, KE}

    B, B_test = sample_basisfunctions(op, test_functions, trial_functions, testqp, trialqp)

    return FMMMatrixSL(
        fmm,
        fmm_t,
        op,
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
        B_test = dropzeros(sparse(rc_test[:, 2], rc_test[:, 1], vals_test))
    else
        B_test = sparse(transpose(B))
    end

    return B, B_test
end
