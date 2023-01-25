using ExaFMMt
using LinearAlgebra
using LinearMaps
using SparseArrays

struct FMMMatrixSL{I, F <: Real, K} <: LinearMaps.LinearMap{K}
    fmm::ExaFMMt.ExaFMM{K}
    B_test::SparseMatrixCSC{F, I}
    Bt_trial::SparseMatrixCSC{F, I}
    BtCB::SparseMatrixCSC{K, I}
    fullmat::SparseMatrixCSC{K, I}
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

    y .= A.Bt_trial * conj.(A.fmm*conj.(A.B_test*x))[:,1] - A.BtCB*x + A.fullmat*x

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

    y .= A.Bt_trial * conj.(A.fmm*conj.(A.B_test*x))[:,1] - A.BtCB*x + A.fullmat*x

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

    y .= A.Bt_trial * conj.(A.fmm*conj.(A.B_test*x))[:,1] - A.BtCB*x + A.fullmat*x

    return y
end