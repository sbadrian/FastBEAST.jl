using ExaFMMt
using LinearAlgebra
using LinearMaps
using SparseArrays

struct FMMMatrixMWDL{I, F <: Real, K} <: LinearMaps.LinearMap{K}
    fmm::ExaFMMt.ExaFMM{K}
    B1::SparseMatrixCSC{F, I}
    B2::SparseMatrixCSC{F, I}
    B3::SparseMatrixCSC{F, I}
    B1t::SparseMatrixCSC{F, I}
    B2t::SparseMatrixCSC{F, I}
    B3t::SparseMatrixCSC{F, I}
    BtCB::SparseMatrixCSC{K, I}
    fullmat::SparseMatrixCSC{K, I}
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

    res1 = conj.(A.fmm*conj.(A.B1*x))[:,2:4]
    res2 = conj.(A.fmm*conj.(A.B2*x))[:,2:4]
    res3 = conj.(A.fmm*conj.(A.B3*x))[:,2:4]

    y1 = A.B1t * (res3[:,2] - res2[:,3])
    y2 = A.B2t * (res1[:,3] - res3[:,1])
    y3 = A.B3t * (res2[:,1] - res1[:,2])

    y.= (y1 + y2 + y3) - A.BtCB*x + A.fullmat*x
    
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

    res1 = conj.(A.fmm*conj.(A.B1*x))[:,2:4]
    res2 = conj.(A.fmm*conj.(A.B2*x))[:,2:4]
    res3 = conj.(A.fmm*conj.(A.B3*x))[:,2:4]

    y1 = A.B1t * (res3[:,2] - res2[:,3])
    y2 = A.B2t * (res1[:,3] - res3[:,1])
    y3 = A.B3t * (res2[:,1] - res1[:,2])

    y.= (y1 + y2 + y3) - A.BtCB*x + A.fullmat*x

    return y
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

    res1 = conj.(A.fmm*conj.(A.B1*x))[:,2:4]
    res2 = conj.(A.fmm*conj.(A.B2*x))[:,2:4]
    res3 = conj.(A.fmm*conj.(A.B3*x))[:,2:4]

    y1 = A.B1t * (res3[:,2] - res2[:,3])
    y2 = A.B2t * (res1[:,3] - res3[:,1])
    y3 = A.B3t * (res2[:,1] - res1[:,2])

    y.= (y1 + y2 + y3) - A.BtCB*x + A.fullmat*x

    return y
end