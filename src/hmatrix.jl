using LinearAlgebra
using LinearMaps
using ProgressMeter

struct HMatrix{I, K} <: LinearMaps.LinearMap{K}
    fullrankblocks::Vector{MatrixBlock{I, K, Matrix{K}}}
    lowrankblocks::Vector{MatrixBlock{I, K, LowRankMatrix{K}}}
    rowdim::I
    columndim::I
    nnz::I
    maxrank::I
    ismultithreaded::Bool
end

function nnz(hmat::HT) where HT <: HMatrix
    return hmat.nnz
end

function compressionrate(hmat::HT) where HT <: HMatrix
    fullsize = hmat.rowdim*hmat.columndim
    return (fullsize - nnz(hmat))/fullsize
end

function ismultithreaded(hmat::HT) where HT <: HMatrix
    return hmat.ismultithreaded
end

function Base.size(hmat::HMatrix, dim=nothing)
    if dim === nothing
        return (hmat.rowdim, hmat.columndim)
    elseif dim == 1
        return hmat.rowdim
    elseif dim == 2
        return hmat.columndim
    else
        error("dim must be either 1 or 2")
    end
end

function Base.size(hmat::Adjoint{T}, dim=nothing) where T <: HMatrix
    if dim === nothing
        return reverse(size(adjoint(hmat)))
    elseif dim == 1
        return size(adjoint(hmat),2)
    elseif dim == 2
        return size(adjoint(hmat),1)
    else
        error("dim must be either 1 or 2")
    end
end

@views function LinearAlgebra.mul!(y::AbstractVecOrMat, A::HMatrix, x::AbstractVector)
    LinearMaps.check_dim_mul(y, A, x)

    fill!(y, zero(eltype(y)))
    
    if !ismultithreaded(A)

        c = zeros(eltype(y), size(A, 1))

        for mb in A.fullrankblocks
            mul!(c[1:size(mb.M,1)], mb.M, x[mb.σ])
            y[mb.τ] .+= c[1:size(mb.M,1)]
        end
        
        for mb in A.lowrankblocks
            mul!(c[1:size(mb.M, 1)], mb.M, x[mb.σ])
            y[mb.τ] .+= c[1:size(mb.M,1)]
        end

    else
        cc = zeros(eltype(y), size(A, 1), Threads.nthreads())
        yy = zeros(eltype(y), size(A, 1), Threads.nthreads())

        Threads.@threads for mb in A.fullrankblocks
            mul!(cc[1:size(mb.M,1), Threads.threadid()], mb.M, x[mb.σ])
            yy[mb.τ, Threads.threadid()] .+= cc[1:size(mb.M,1), Threads.threadid()]
        end
        
        Threads.@threads for mb in A.lowrankblocks
            mul!(cc[1:size(mb.M, 1), Threads.threadid()], mb.M, x[mb.σ])
            yy[mb.τ, Threads.threadid()] .+= cc[1:size(mb.M,1), Threads.threadid()]
        end

        y[:] = sum(yy, dims=2)
    end

    return y
end


@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    transA::LinearMaps.TransposeMap{<:Any,<:HMatrix},
    x::AbstractVector
)
    LinearMaps.check_dim_mul(y, transA, x)

    fill!(y, zero(eltype(y)))

    if !ismultithreaded(transA.lmap)

        c = zeros(eltype(y), size(transA,1))

        for mb in transA.lmap.fullrankblocks
            mul!(c[1:size(mb.M, 2)], transpose(mb.M), x[mb.τ])
            y[mb.σ] .+= c[1:size(mb.M, 2)]
        end

        for mb in transA.lmap.lowrankblocks
            mul!(c[1:size(mb.M,2)], transpose(mb.M), x[mb.τ])
            y[mb.σ] .+= c[1:size(mb.M,2)]
        end

    else

        cc = zeros(eltype(y), size(transA, 1), Threads.nthreads())
        yy = zeros(eltype(y), size(transA, 1), Threads.nthreads())

        Threads.@threads for mb in transA.lmap.fullrankblocks
            mul!(cc[1:size(mb.M, 2), Threads.threadid()], transpose(mb.M), x[mb.τ])
            yy[mb.σ, Threads.threadid()] .+= cc[1:size(mb.M, 2), Threads.threadid()]
        end
        
        Threads.@threads for mb in transA.lmap.lowrankblocks
            mul!(cc[1:size(mb.M, 2), Threads.threadid()], transpose(mb.M), x[mb.τ])
            yy[mb.σ, Threads.threadid()] .+= cc[1:size(mb.M, 2), Threads.threadid()]
        end

        y[:] = sum(yy, dims=2)

    end

    return y
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    transA::LinearMaps.AdjointMap{<:Any,<:HMatrix},
    x::AbstractVector
)
    LinearMaps.check_dim_mul(y, transA, x)

    fill!(y, zero(eltype(y)))

    if !ismultithreaded(transA.lmap)

        c = zeros(eltype(y), size(transA,1))

        for mb in transA.lmap.fullrankblocks
            mul!(c[1:size(adjoint(mb.M),1)], adjoint(mb.M), x[mb.τ])
            y[mb.σ] .+= c[1:size(mb.M, 2)]
        end

        for mb in transA.lmap.lowrankblocks
            mul!(c[1:size(adjoint(mb.M),1)], adjoint(mb.M), x[mb.τ])
            y[mb.σ] .+= c[1:size(mb.M,2)]
        end

    else

        cc = zeros(eltype(y), size(transA, 1), Threads.nthreads())
        yy = zeros(eltype(y), size(transA, 1), Threads.nthreads())

        Threads.@threads for mb in transA.lmap.fullrankblocks
            mul!(cc[1:size(mb.M, 2), Threads.threadid()], transpose(mb.M), x[mb.τ])
            yy[mb.σ, Threads.threadid()] .+= cc[1:size(mb.M, 2), Threads.threadid()]
        end
        
        Threads.@threads for mb in transA.lmap.lowrankblocks
            mul!(cc[1:size(mb.M, 2), Threads.threadid()], transpose(mb.M), x[mb.τ])
            yy[mb.σ, Threads.threadid()] .+= cc[1:size(mb.M, 2), Threads.threadid()]
        end

        y[:] = sum(yy, dims=2)

    end

    return y
end

function HMatrix(
    matrixassembler::Function,
    testtree::T,#::BoxTreeNode,
    sourcetree::T,#::BoxTreeNode,
    ::Type{I},
    ::Type{K};
    farmatrixassembler=matrixassembler,
    compressor=ACAOptions(),
    multithreading=false,
    verbose=false
) where {I, K, T <: AbstractNode}
    
    fullinteractions = SVector{2}[]
    compressableinteractions = SVector{2}[]
    
    computerinteractions!(
        testtree,
        sourcetree,
        fullinteractions,
        compressableinteractions
    )

    MBF = MatrixBlock{I, K, Matrix{K}}
    fullrankblocks_perthread = Vector{MBF}[]
    fullrankblocks = MBF[]

    rowdim = numindices(testtree)
    coldim = numindices(sourcetree)

    if !multithreading
        am = allocate_aca_memory(K, rowdim, coldim, maxrank=compressor.maxrank)
    else
        ams = ACAGlobalMemory{I, real(K), K}[]
        for i in 1:Threads.nthreads()
            push!(ams, allocate_aca_memory(K, rowdim, coldim, maxrank=compressor.maxrank))
        end
    end
    nonzeros_perthread = I[]
    nonzeros = 0
    verbose && println("Number of full interactions: ", length(fullinteractions))
    verbose && println(
        "Number of compressable interactions: ",
        length(compressableinteractions)
    )

    if verbose
        p = Progress(length(fullinteractions), desc="Computing full interactions: ")
    end

    if !multithreading
        for fullinteraction in fullinteractions
            nonzeros += numindices(fullinteraction[1])*numindices(fullinteraction[2])
            push!(
                fullrankblocks,
                getfullmatrixview(
                    matrixassembler,
                    fullinteraction[1],
                    fullinteraction[2],
                    I,
                    K
                )
            )
            verbose && next!(p)
        end
    elseif multithreading
        for i in 1:Threads.nthreads()
            push!(fullrankblocks_perthread, MBF[])
            push!(nonzeros_perthread, 0)
        end

        Threads.@threads for fullinteraction in fullinteractions
            nonzeros_perthread[Threads.threadid()] += 
                numindices(fullinteraction[1])*numindices(fullinteraction[2])
            push!(
                fullrankblocks_perthread[Threads.threadid()],
                getfullmatrixview(
                    matrixassembler,
                    fullinteraction[1],
                    fullinteraction[2],
                    I,
                    K
                )
            )
            verbose && next!(p)
        end

        for i in eachindex(fullrankblocks_perthread)
            append!(fullrankblocks, fullrankblocks_perthread[i])
        end
    end

    MBL = MatrixBlock{I, K, LowRankMatrix{K}}
    lowrankblocks_perthread = Vector{MBL}[]
    lowrankblocks = MBL[]

    if verbose
        p = Progress(length(compressableinteractions), desc="Compressing far interactions: ")
    end

    if !multithreading
        for compressableinteraction in compressableinteractions
            push!(
                lowrankblocks, 
                getcompressedmatrix(
                    farmatrixassembler,
                    compressableinteraction[1],
                    compressableinteraction[2],
                    I,
                    K,
                    am,
                    compressor=compressor
                )
            )
            nonzeros += nnz(lowrankblocks[end])
            verbose && next!(p)
        end
    elseif multithreading
        for i in 1:Threads.nthreads()
            push!(lowrankblocks_perthread, MBL[])
        end

        Threads.@threads for compressableinteraction in compressableinteractions
            push!(
                lowrankblocks_perthread[Threads.threadid()],
                getcompressedmatrix(
                    farmatrixassembler,
                    compressableinteraction[1],
                    compressableinteraction[2],
                    I,
                    K,
                    ams[Threads.threadid()],
                    compressor=compressor
                )
            )
            nonzeros_perthread[Threads.threadid()] += 
                nnz(lowrankblocks_perthread[Threads.threadid()][end])
            verbose && next!(p)
        end

        for i in eachindex(lowrankblocks_perthread)
            append!(lowrankblocks, lowrankblocks_perthread[i])
        end
    end

    return HMatrix{I, K}(
        fullrankblocks,
        lowrankblocks,
        rowdim,
        coldim,
        nonzeros + sum(nonzeros_perthread),
        compressor.maxrank,
        multithreading
    )
end

function computerinteractions!(
    testnode::T,
    sourcenode::T,
    fullinteractions::Vector{SVector{2}},
    compressableinteractions::Vector{SVector{2}}
) where T <: AbstractNode
    if iscompressable(sourcenode, testnode)
        if level(sourcenode) == 0 && level(testnode) == 0
            push!(compressableinteractions, SVector(testnode, sourcenode))
            return
        else
            error("We do not expect this behavior")
        end
    end

    if !haschildren(sourcenode) && !haschildren(testnode)
        push!(fullinteractions, SVector(testnode, sourcenode))
        return
    else
        if !haschildren(sourcenode)
            schild = sourcenode
            for tchild in children(testnode)
                decide_compression(
                    tchild, 
                    schild, 
                    fullinteractions, 
                    compressableinteractions
                )
            end
        elseif !haschildren(testnode)
            tchild = testnode
            for schild in children(sourcenode)
                decide_compression(
                    tchild, 
                    schild, 
                    fullinteractions, 
                    compressableinteractions
                )
            end
        else
            for schild in children(sourcenode)
                for tchild in children(testnode)
                    decide_compression(
                        tchild, 
                        schild, 
                        fullinteractions, 
                        compressableinteractions
                    )
                end
            end
        end
    end
end

function decide_compression(ttchild, sschild, fullinteractions, compressableinteractions)
    if numindices(ttchild) > 0 && numindices(sschild) > 0
        if iscompressable(ttchild, sschild)
            push!(compressableinteractions, SVector(ttchild, sschild))

            return
        else
            computerinteractions!(
                ttchild,
                sschild,
                fullinteractions,
                compressableinteractions
            )
        end
    end
end

function getfullmatrixview(
    matrixassembler,
    testnode,
    sourcenode,
    ::Type{I},
    ::Type{K};
) where {I, K}
    matrix = zeros(K, numindices(testnode), numindices(sourcenode))
    matrixassembler(matrix, indices(testnode), indices(sourcenode))

    return MatrixBlock{I, K, Matrix{K}}(
        matrix,
        indices(testnode),
        indices(sourcenode)
    )
end

function getcompressedmatrix(
    matrixassembler::Function,
    testnode,
    sourcenode,
    ::Type{I},
    ::Type{K},
    am;
    compressor=ACAOptions()
) where {I, K}

        lm = LazyMatrix(matrixassembler, indices(testnode), indices(sourcenode), K)

        U, V = aca(
            lm,
            am;
            rowpivstrat=compressor.rowpivstrat,
            columnpivstrat=compressor.columnpivstrat,
            tol=compressor.tol,
            svdrecompress=compressor.svdrecompress
        )

        mbl = MatrixBlock{I, K, LowRankMatrix{K}}(
            LowRankMatrix(U, V),
            indices(testnode),
            indices(sourcenode)
        )

    return mbl
end

##

real(ComplexF64)