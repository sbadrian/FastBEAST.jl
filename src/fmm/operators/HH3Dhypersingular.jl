using ExaFMMt
using LinearAlgebra
using LinearMaps
using SparseArrays

struct FMMMatrixHS{I, F <: Real, K} <: LinearMaps.LinearMap{K}
    fmm::ExaFMMt.ExaFMM{K}
    normals_test::Matrix{F}
    normals_trial::Matrix{F}
    Bcurl1_test::SparseMatrixCSC{F, I}
    Bcurl2_test::SparseMatrixCSC{F, I}
    Bcurl3_test::SparseMatrixCSC{F, I}
    Btcurl1_trial::SparseMatrixCSC{F, I}
    Btcurl2_trial::SparseMatrixCSC{F, I}
    Btcurl3_trial::SparseMatrixCSC{F, I}
    B_test::SparseMatrixCSC{F, I}
    Bt_trial::SparseMatrixCSC{F, I}
    BtCB::SparseMatrixCSC{K, I}
    fullmat::SparseMatrixCSC{K, I}
    rowdim::I
    columndim::I
end

function Base.size(fmat::FMMMatrixHS, dim=nothing)
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

function Base.size(fmat::Adjoint{T}, dim=nothing) where T <: FMMMatrixHS
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

@views function LinearAlgebra.mul!(y::AbstractVecOrMat, A::FMMMatrixHS, x::AbstractVector)
    LinearMaps.check_dim_mul(y, A, x)

    if eltype(x) != eltype(A)
        x = eltype(A).(x)
    end
    fill!(y, zero(eltype(y)))

    fmm_curl1 = A.Btcurl1_trial * conj.(A.fmm * conj.(A.Bcurl1_test * x))[:,1]
    fmm_curl2 = A.Btcurl2_trial * conj.(A.fmm * conj.(A.Bcurl2_test * x))[:,1]
    fmm_curl3 = A.Btcurl3_trial * conj.(A.fmm * conj.(A.Bcurl3_test * x))[:,1]

    y1 = fmm_curl1 + fmm_curl2 + fmm_curl3

    fmm_res1 = A.normals_trial[:,1] .* conj.(
        A.fmm*conj.(A.normals_test[:,1] .* (A.B_test * x))
    )[:,1]
    fmm_res2 = A.normals_trial[:,2] .* conj.(
        A.fmm*conj.(A.normals_test[:,2] .* (A.B_test * x))
    )[:,1]
    fmm_res3 = A.normals_trial[:,3] .* conj.(
        A.fmm*conj.(A.normals_test[:,3] .* (A.B_test * x))
    )[:,1]

    y2 = A.fmm.fmmoptions.wavek^2 * A.Bt_trial * (fmm_res1 + fmm_res2 + fmm_res3)

    y .= (y1 - y2) - A.BtCB*x + A.fullmat*x

    return y
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    A::LinearMaps.TransposeMap{<:Any,<:FMMMatrixHS},
    x::AbstractVector
)
    LinearMaps.check_dim_mul(y, A, x)

    if eltype(x) != eltype(A)
        x = eltype(A).(x)
    end
    fill!(y, zero(eltype(y)))

    fmm_curl1 = A.Btcurl1_trial * conj.(A.fmm * conj.(A.Bcurl1_test * x))[:,1]
    fmm_curl2 = A.Btcurl2_trial * conj.(A.fmm * conj.(A.Bcurl2_test * x))[:,1]
    fmm_curl3 = A.Btcurl3_trial * conj.(A.fmm * conj.(A.Bcurl3_test * x))[:,1]

    y1 = fmm_curl1 + fmm_curl2 + fmm_curl3

    fmm_res1 = A.normals_trial[:,1] .* conj.(
        A.fmm*conj.(A.normals_test[:,1] .* (A.B_test * x))
    )[:,1]
    fmm_res2 = A.normals_trial[:,2] .* conj.(
        A.fmm*conj.(A.normals_test[:,2] .* (A.B_test * x))
    )[:,1]
    fmm_res3 = A.normals_trial[:,3] .* conj.(
        A.fmm*conj.(A.normals_test[:,3] .* (A.B_test * x))
    )[:,1]

    y2 = A.fmm.fmmoptions.wavek^2 * A.Bt_trial * (fmm_res1 + fmm_res2 + fmm_res3)

    y .= (y1 - y2) - A.BtCB*x + A.fullmat*x

    return y
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    A::LinearMaps.AdjointMap{<:Any,<:FMMMatrixHS},
    x::AbstractVector
)
    LinearMaps.check_dim_mul(y, A, x)

    if eltype(x) != eltype(A)
        x = eltype(A).(x)
    end
    fill!(y, zero(eltype(y)))

    fmm_curl1 = A.Btcurl1_trial * conj.(A.fmm * conj.(A.Bcurl1_test * x))[:,1]
    fmm_curl2 = A.Btcurl2_trial * conj.(A.fmm * conj.(A.Bcurl2_test * x))[:,1]
    fmm_curl3 = A.Btcurl3_trial * conj.(A.fmm * conj.(A.Bcurl3_test * x))[:,1]

    y1 = fmm_curl1 + fmm_curl2 + fmm_curl3

    fmm_res1 = A.normals_trial[:,1] .* conj.(
        A.fmm*conj.(A.normals_test[:,1] .* (A.B_test * x))
    )[:,1]
    fmm_res2 = A.normals_trial[:,2] .* conj.(
        A.fmm*conj.(A.normals_test[:,2] .* (A.B_test * x))
    )[:,1]
    fmm_res3 = A.normals_trial[:,3] .* conj.(
        A.fmm*conj.(A.normals_test[:,3] .* (A.B_test * x))
    )[:,1]

    y2 = A.fmm.fmmoptions.wavek^2 * A.Bt_trial * (fmm_res1 + fmm_res2 + fmm_res3)

    y .= (y1 - y2) - A.BtCB*x + A.fullmat*x

    return y
end