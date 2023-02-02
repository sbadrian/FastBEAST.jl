using BEAST
using ExaFMMt
using LinearAlgebra
using LinearMaps
using SparseArrays


struct FMMMatrixADL{I, F <: Real, K} <: LinearMaps.LinearMap{K}
    fmm::ExaFMMt.ExaFMM{K}
    normals::Matrix{F}
    B_trial::SparseMatrixCSC{F, I}
    Bt_test::SparseMatrixCSC{F, I}
    BtCB::HMatrix{I, K}
    fullmat::HMatrix{I, K}
    rowdim::I
    columndim::I
end

function Base.size(fmat::FMMMatrixADL, dim=nothing)
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

function Base.size(fmat::Adjoint{T}, dim=nothing) where T <: FMMMatrixADL
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

@views function LinearAlgebra.mul!(y::AbstractVecOrMat, A::FMMMatrixADL, x::AbstractVector)
    LinearMaps.check_dim_mul(y, A, x)

    if eltype(x) != eltype(A)
        x = eltype(A).(x)
    end
    fill!(y, zero(eltype(y)))

    res = A.fmm*conj.(A.B_trial * x)
    fmm_res1 = A.normals[:,1] .* conj.(res)[:,2]
    fmm_res2 = A.normals[:,2] .* conj.(res)[:,3]
    fmm_res3 = A.normals[:,3] .* conj.(res)[:,4]
    y.= A.Bt_test * (fmm_res1 + fmm_res2 + fmm_res3) - A.BtCB*x + A.fullmat*x
    
    return y
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    A::LinearMaps.TransposeMap{<:Any,<:FMMMatrixADL},
    x::AbstractVector
)
    LinearMaps.check_dim_mul(y, A, x)

    if eltype(x) != eltype(A)
        x = eltype(A).(x)
    end
    fill!(y, zero(eltype(y)))

    res = A.fmm*conj.(A.B_trial * x)
    fmm_res1 = A.normals[:,1] .* conj.(res)[:,2]
    fmm_res2 = A.normals[:,2] .* conj.(res)[:,3]
    fmm_res3 = A.normals[:,3] .* conj.(res)[:,4]
    y.= A.Bt_test * (fmm_res1 + fmm_res2 + fmm_res3) - A.BtCB*x + A.fullmat*x

    return y
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    A::LinearMaps.AdjointMap{<:Any,<:FMMMatrixADL},
    x::AbstractVector
)
    LinearMaps.check_dim_mul(y, A, x)

    if eltype(x) != eltype(A)
        x = eltype(A).(x)
    end

    fill!(y, zero(eltype(y)))

    res = A.fmm*conj.(A.B_trial * x)
    fmm_res1 = A.normals[:,1] .* conj.(res)[:,2]
    fmm_res2 = A.normals[:,2] .* conj.(res)[:,3]
    fmm_res3 = A.normals[:,3] .* conj.(res)[:,4]
    y.= A.Bt_test * (fmm_res1 + fmm_res2 + fmm_res3) - A.BtCB*x + A.fullmat*x

    return y
end

function FMMMatrix(
    op::BEAST.HH3DDoubleLayerTransposed,
    test_functions::BEAST.Space, 
    trial_functions::BEAST.Space, 
    testqp::Matrix,
    trialqp::Matrix,
    fmm::ExaFMMt.ExaFMM{K},
    BtCB::HMatrix{I, K},
    fullmat::HMatrix{I, K}
) where {I, K}

    B, normals, B_test = sample_basisfunctions(op, test_functions, trial_functions, testqp, trialqp)

    return FMMMatrixADL(
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
    op::BEAST.HH3DDoubleLayerTransposed,
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
        B_test = dropzeros(sparse(rc_test[:, 2], rc_test[:, 1], vals_test))
    else
        B_test = sparse(transpose(B))
    end

    return B, normals, B_test
end