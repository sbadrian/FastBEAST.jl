using BEAST
using ExaFMMt
using LinearAlgebra
using LinearMaps
using SparseArrays


struct FMMMatrixHS{I, F <: Real, K, KE} <: LinearMaps.LinearMap{K}
    fmm::ExaFMMt.ExaFMM{KE}
    fmm_t::ExaFMMt.ExaFMM{KE}
    op::BEAST.HH3DHyperSingularFDBIO
    normals_trial::Matrix{F}
    normals_test::Matrix{F}
    B1curl_trial::SparseMatrixCSC{F, I}
    B2curl_trial::SparseMatrixCSC{F, I}
    B3curl_trial::SparseMatrixCSC{F, I}
    B1curl_test::SparseMatrixCSC{F, I}
    B2curl_test::SparseMatrixCSC{F, I}
    B3curl_test::SparseMatrixCSC{F, I}
    B_trial::SparseMatrixCSC{F, I}
    B_test::SparseMatrixCSC{F, I}
    BtCB::HMatrix{I, K}
    fullmat::HMatrix{I, K}
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

    if A.op.alpha != 0.0
        fmm_res1 = A.normals_test[:,1] .* (
            A.fmm * (A.normals_trial[:,1] .* (A.B_trial * xfmm))
        )[:,1]
        fmm_res2 = A.normals_test[:,2] .* (
            A.fmm * (A.normals_trial[:,2] .* (A.B_trial * xfmm))
        )[:,1]
        fmm_res3 = A.normals_test[:,3] .* (
            A.fmm * (A.normals_trial[:,3] .* (A.B_trial * xfmm))
        )[:,1]

        y .+= A.op.alpha * A.B_test * (fmm_res1 + fmm_res2 + fmm_res3)
    end

    if A.op.beta != 0.0
        fmm_curl1 = A.B1curl_test * (A.fmm * (A.B1curl_trial * xfmm))[:,1]
        fmm_curl2 = A.B2curl_test * (A.fmm * (A.B2curl_trial * xfmm))[:,1]
        fmm_curl3 = A.B3curl_test * (A.fmm * (A.B3curl_trial * xfmm))[:,1]
    
        y .+= A.op.beta .* (fmm_curl1 + fmm_curl2 + fmm_curl3)
    end

    y .+= - A.BtCB*x + A.fullmat*x

    return y
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    At::LinearMaps.TransposeMap{<:Any,<:FMMMatrixHS},
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

    xx = transpose(A.B_test)*xfmm

    if A.op.alpha != 0.0
        fmm_res1 = A.normals_trial[:,1] .* (
            A.fmm_t * (A.normals_test[:,1] .* (xx))
        )[:,1]
        fmm_res2 = A.normals_trial[:,2] .* (
            A.fmm_t * (A.normals_test[:,2] .* (xx))
        )[:,1]
        fmm_res3 = A.normals_trial[:,3] .* (
            A.fmm_t * (A.normals_test[:,3] .* (xx))
        )[:,1]
    
        y .+= A.op.alpha * transpose(A.B_trial) * (fmm_res1 + fmm_res2 + fmm_res3)    
    end

    if A.op.beta != 0.0
        fmm_curl1 = transpose(A.B1curl_trial) * (A.fmm_t * (transpose(A.B1curl_test) * xfmm))[:,1]
        fmm_curl2 = transpose(A.B2curl_trial) * (A.fmm_t * (transpose(A.B2curl_test) * xfmm))[:,1]
        fmm_curl3 = transpose(A.B3curl_trial) * (A.fmm_t * (transpose(A.B3curl_test) * xfmm))[:,1]

        y .+= A.op.beta .* (fmm_curl1 + fmm_curl2 + fmm_curl3)
    end

    y .+= - transpose(A.BtCB)*x + transpose(A.fullmat)*x

    return y
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    At::LinearMaps.AdjointMap{<:Any,<:FMMMatrixHS},
    x::AbstractVector
)

    mul!(y, transpose(adjoint(At)), conj(x))

    return conj!(y)
end

function FMMMatrix(
    op::BEAST.HH3DHyperSingularFDBIO,
    test_functions::BEAST.Space, 
    trial_functions::BEAST.Space, 
    testqp::Matrix,
    trialqp::Matrix,
    fmm::ExaFMMt.ExaFMM{KE},
    fmm_t::ExaFMMt.ExaFMM{KE},
    BtCB::HMatrix{I, K},
    fullmat::HMatrix{I, K},
) where {I, K, KE}

    normals_trial, B1curl, B2curl, B3curl, B, normals_test,
        B1curl_test, B2curl_test, B3curl_test, B_test = sample_basisfunctions(
            op,
            test_functions,
            trial_functions,
            testqp,
            trialqp
        )

    return FMMMatrixHS(
        fmm,
        fmm_t,
        op,
        normals_trial,
        normals_test,
        B1curl,
        B2curl,
        B3curl,
        B1curl_test,
        B2curl_test,
        B3curl_test,
        B,
        B_test,
        BtCB,
        fullmat,
        size(fullmat)[1],
        size(fullmat)[2]
    )

end  

function sample_basisfunctions(
    op::BEAST.HH3DHyperSingularFDBIO,
    test_functions::BEAST.Space, 
    trial_functions::BEAST.Space, 
    testqp::Matrix,
    trialqp::Matrix,
)

    normals_trial = getnormals(trialqp)
    normals_test = getnormals(testqp)
    rc_curl, vals_curl = sample_curlbasisfunctions(trialqp, trial_functions)
    B1curl = dropzeros(sparse(rc_curl[:, 1], rc_curl[:, 2], vals_curl[:, 1]))
    B2curl = dropzeros(sparse(rc_curl[:, 1], rc_curl[:, 2], vals_curl[:, 2]))
    B3curl = dropzeros(sparse(rc_curl[:, 1], rc_curl[:, 2], vals_curl[:, 3]))
    B1curl_test, B2curl_test, B3curl_test = B1curl, B2curl, B3curl

    rc, vals = sample_basisfunctions(op, trialqp, trial_functions)
    B = dropzeros(sparse(rc[:, 1], rc[:, 2], vals))
    B_test = B

    if test_functions != trial_functions
        normals_test = getnormals(testqp)
        rc_curl, vals_curl = sample_curlbasisfunctions(testqp, test_functions)
        B1curl_test = dropzeros(sparse(rc_curl[:, 2], rc_curl[:, 1], vals_curl[:, 1]))
        B2curl_test = dropzeros(sparse(rc_curl[:, 2], rc_curl[:, 1], vals_curl[:, 2]))
        B3curl_test = dropzeros(sparse(rc_curl[:, 2], rc_curl[:, 1], vals_curl[:, 3]))
        
        rc, vals = sample_basisfunctions(op, testqp, test_functions)
        B_test = dropzeros(sparse(rc[:, 2], rc[:, 1], vals))
    else
        B1curl_test = sparse(transpose(B1curl))
        B2curl_test = sparse(transpose(B2curl))
        B3curl_test = sparse(transpose(B3curl)) 
        B_test = sparse(transpose(B))
    end

    return normals_trial, B1curl, B2curl, B3curl, B,
        normals_test, B1curl_test, B2curl_test, B3curl_test, B_test
end