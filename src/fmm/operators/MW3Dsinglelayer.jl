using ExaFMMt
using LinearAlgebra
using LinearMaps
using SparseArrays

struct FMMMatrixMWSL{I, F <: Real, K} <: LinearMaps.LinearMap{K}
    fmm::ExaFMMt.ExaFMM{K}
    B1::SparseMatrixCSC{F, I}
    B2::SparseMatrixCSC{F, I}
    B3::SparseMatrixCSC{F, I}
    Bdiv::SparseMatrixCSC{F, I}
    B1t::SparseMatrixCSC{F, I}
    B2t::SparseMatrixCSC{F, I}
    B3t::SparseMatrixCSC{F, I}
    Bdivt::SparseMatrixCSC{F, I}
    BtCB::SparseMatrixCSC{K, I}
    fullmat::SparseMatrixCSC{K, I}
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

    res1 = A.B1t*conj.(A.fmm*conj.(A.B1*x))[:,1]
    res2 = A.B2t*conj.(A.fmm*conj.(A.B2*x))[:,1]
    res3 = A.B3t*conj.(A.fmm*conj.(A.B3*x))[:,1]

    y1 = -(im*A.fmm.fmmoptions.wavek .* (res1 + res2 + res3))

    y2 = 1 / (im*A.fmm.fmmoptions.wavek) .* (A.Bdivt * conj.(A.fmm * conj.(A.Bdiv*x))[:,1])

    y.= (y1 - y2) - A.BtCB*x + A.fullmat*x
    
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

    res1 = B1t*conj.(A.fmm*conj.(A.B1*x))[:,1]
    res2 = B2t*conj.(A.fmm*conj.(A.B2*x))[:,1]
    res3 = B3t*conj.(A.fmm*conj.(A.B3*x))[:,1]

    y1 = -(im*A.fmm.fmmoptions.wavek .* (res1 + res2 + res3))

    y2 = 1 / (im*A.fmm.fmmoptions.wavek) .* (A.Bdivt * conj.(A.fmm * conj.(A.Bdiv*x))[:,1])

    y.= (y1 - y2) - A.BtCB*x + A.fullmat*x

    return y
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    A::LinearMaps.AdjointMap{<:Any,<:FMMMatrixMWSL},
    x::AbstractVector
)
    LinearMaps.check_dim_mul(y, A, x)

    if eltype(x) != eltype(A)
        x = eltype(A).(x)
    end
    fill!(y, zero(eltype(y)))

    res1 = B1t*conj.(A.fmm*conj.(A.B1*x))[:,1]
    res2 = B2t*conj.(A.fmm*conj.(A.B2*x))[:,1]
    res3 = B3t*conj.(A.fmm*conj.(A.B3*x))[:,1]

    y1 = -(im*A.fmm.fmmoptions.wavek .* (res1 + res2 + res3))

    y2 = 1 / (im*A.fmm.fmmoptions.wavek) .* (A.Bdivt * conj.(A.fmm * conj.(A.Bdiv*x))[:,1])

    y.= (y1 - y2) - A.BtCB*x + A.fullmat*x

    return y
end