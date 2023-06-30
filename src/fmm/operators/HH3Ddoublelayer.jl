using BEAST
using ExaFMMt
using LinearAlgebra
using LinearMaps
using SparseArrays


struct FMMMatrixDL{I, F <: Real, K} <: LinearMaps.LinearMap{K}
    fmm::ExaFMMt.ExaFMM{K}
    normals::Matrix{F}
    B_trial::SparseMatrixCSC{F, I}
    B_test::SparseMatrixCSC{F, I}
    BtCB::HMatrix{I, K}
    fullmat::HMatrix{I, K}
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

    a = A.normals[:,1] .* (A.B_trial * x)
    b = A.normals[:,2] .* (A.B_trial * x)
    c = A.normals[:,3] .* (A.B_trial * x)

    aa = conj.(A.fmm*conj.(a))[:,1]
    bb = conj.(A.fmm*conj.(b))[:,2]
    cc = conj.(A.fmm*conj.(c))[:,3]

    @show size(A.B_test)
    @show size(aa)
    fmm_res1 = A.B_test * aa
    fmm_res2 = A.B_test * bb
    fmm_res3 = A.B_test * cc
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

    fmm_res1 = A.B_test * conj.(A.fmm*conj.(A.normals[:,1] .* (A.B_trial * x)))[:,2]
    fmm_res2 = A.B_test * conj.(A.fmm*conj.(A.normals[:,2] .* (A.B_trial * x)))[:,3]
    fmm_res3 = A.B_test * conj.(A.fmm*conj.(A.normals[:,3] .* (A.B_trial * x)))[:,4]
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

    fmm_res1 = A.B_test * conj.(A.fmm*conj.(A.normals[:,1] .* (A.B_trial * x)))[:,2]
    fmm_res2 = A.B_test * conj.(A.fmm*conj.(A.normals[:,2] .* (A.B_trial * x)))[:,3]
    fmm_res3 = A.B_test * conj.(A.fmm*conj.(A.normals[:,3] .* (A.B_trial * x)))[:,4]
    fmm_res = -(fmm_res1 + fmm_res2 + fmm_res3)
    y .= fmm_res - A.BtCB*x + A.fullmat*x

    return y
end

function FMMMatrix(
    op::BEAST.HH3DDoubleLayerFDBIO,
    test_functions::BEAST.Space, 
    trial_functions::BEAST.Space, 
    testqp::Matrix,
    trialqp::Matrix,
    fmm::ExaFMMt.ExaFMM{K},
    BtCB::HMatrix{I, K},
    fullmat::HMatrix{I, K}
) where {I, K}

    B, normals, B_test = sample_basisfunctions(
        op,
        test_functions,
        trial_functions,
        testqp,
        trialqp
    )

    return FMMMatrixDL(
        fmm,
        normals,
        B,
        B_test,
        BtCB,
        fullmat,
        size(fullmat)[1],
        size(fullmat)[2]
    )

end

function sample_basisfunctions(
    op::BEAST.HH3DDoubleLayerFDBIO,
    test_functions::BEAST.Space, 
    trial_functions::BEAST.Space, 
    testqp::Matrix,
    trialqp::Matrix
)   
    normals = getnormals(trialqp)
    rc, vals = sample_basisfunctions(op, trialqp, trial_functions)
    B = dropzeros(sparse(rc[:, 1], rc[:, 2], vals)) 
    B_test = B

    if test_functions != trial_functions 
        rc_test, vals_test = sample_basisfunctions(op, testqp, test_functions)
        B_test = sparse(transpose(dropzeros(sparse(rc_test[:, 1], rc_test[:, 2], vals_test))))
    else
        B_test = sparse(transpose(B))
    end

    return B, normals, B_test
end