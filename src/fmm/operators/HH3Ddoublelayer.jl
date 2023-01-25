using ExaFMMt
using LinearAlgebra
using LinearMaps
using SparseArrays

struct FMMMatrixDL{I, F <: Real, K} <: LinearMaps.LinearMap{K}
    fmm::ExaFMMt.ExaFMM{K}
    normals::Matrix{F}
    B::SparseMatrixCSC{F, I}
    Bt::SparseMatrixCSC{F, I}
    BtCB::SparseMatrixCSC{K, I}
    fullmat::SparseMatrixCSC{K, I}
    rowdim::I
    columndim::I
end

function Base.size(fmat::FMMMatrixDL, dim=nothing)
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

function Base.size(fmat::Adjoint{T}, dim=nothing) where T <: FMMMatrixDL
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

@views function LinearAlgebra.mul!(y::AbstractVecOrMat, A::FMMMatrixDL, x::AbstractVector)
    LinearMaps.check_dim_mul(y, A, x)

    if eltype(x) != eltype(A)
        x = eltype(A).(x)
    end
    fill!(y, zero(eltype(y)))

    fmm_res1 = A.Bt * conj.(A.fmm*conj.(A.normals[:,1] .* (A.B * x)))[:,2]
    fmm_res2 = A.Bt * conj.(A.fmm*conj.(A.normals[:,2] .* (A.B * x)))[:,3]
    fmm_res3 = A.Bt * conj.(A.fmm*conj.(A.normals[:,3] .* (A.B * x)))[:,4]
    fmm_res = -(fmm_res1 + fmm_res2 + fmm_res3)
    y .= fmm_res - A.BtCB*x + A.fullmat*x

    return y
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    A::LinearMaps.TransposeMap{<:Any,<:FMMMatrixDL},
    x::AbstractVector
)
    LinearMaps.check_dim_mul(y, A, x)

    if eltype(x) != eltype(A)
        x = eltype(A).(x)
    end
    fill!(y, zero(eltype(y)))

    fmm_res1 = A.Bt * conj.(A.fmm*conj.(A.normals[:,1] .* (A.B * x)))[:,2]
    fmm_res2 = A.Bt * conj.(A.fmm*conj.(A.normals[:,2] .* (A.B * x)))[:,3]
    fmm_res3 = A.Bt * conj.(A.fmm*conj.(A.normals[:,3] .* (A.B * x)))[:,4]
    fmm_res = -(fmm_res1 + fmm_res2 + fmm_res3)
    y .= fmm_res - A.BtCB*x + A.fullmat*x

    return y
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    A::LinearMaps.AdjointMap{<:Any,<:FMMMatrixDL},
    x::AbstractVector
)
    LinearMaps.check_dim_mul(y, A, x)

    if eltype(x) != eltype(A)
        x = eltype(A).(x)
    end

    fill!(y, zero(eltype(y)))

    fmm_res1 = A.Bt * conj.(A.fmm*conj.(A.normals[:,1] .* (A.B * x)))[:,2]
    fmm_res2 = A.Bt * conj.(A.fmm*conj.(A.normals[:,2] .* (A.B * x)))[:,3]
    fmm_res3 = A.Bt * conj.(A.fmm*conj.(A.normals[:,3] .* (A.B * x)))[:,4]
    fmm_res = -(fmm_res1 + fmm_res2 + fmm_res3)
    y .= fmm_res - A.BtCB*x + A.fullmat*x

    return y
end