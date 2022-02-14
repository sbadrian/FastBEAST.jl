using LinearAlgebra
using LinearMaps
using ProgressMeter

struct HMatrix{I, F} <: LinearMaps.LinearMap{F}
    fullrankblocks::Vector{MatrixBlock{I, F, Matrix{F}}}
    lowrankblocks::Vector{MatrixBlock{I, F, LowRankMatrix{F}}}
    rowdim::I
    columndim::I
    nnz::I
    maxrank::I
    ismultithreaded::Bool
end

function nnz(hmat::HMatrix) where HT <: HMatrix
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

        y = sum(yy, dims=2)
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

        y = sum(yy, dims=2)

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

        y = sum(yy, dims=2)

    end

    return y
end

function HMatrix(
    matrixassembler::Function,
    testtree::N,#::BoxTreeNode,
    sourcetree::N,#::BoxTreeNode,
    ::Type{I},
    ::Type{F};
    compressor=:naive,
    tol=1e-4,
    maxrank=100,
    threading=:single,
    farmatrixassembler=matrixassembler,
    verbose=false,
    svdrecompress=true
) where {I, F, N <: AbstractNode}
    
    fullinteractions = SVector{2}[]
    compressableinteractions = SVector{2}[]
    

    computerinteractions!(
        testtree,
        sourcetree,
        fullinteractions,
        compressableinteractions
    )
    MBF = MatrixBlock{I, F, Matrix{F}}
    fullrankblocks_perthread = Vector{MBF}[]
    fullrankblocks = MBF[]

    rowdim = length(testtree.data)
    coldim = length(sourcetree.data)

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

    if threading == :single
        for fullinteraction in fullinteractions
            nonzeros += length(fullinteraction[1].data)*length(fullinteraction[2].data)
            push!(
                fullrankblocks,
                getfullmatrixview(
                    matrixassembler,
                    fullinteraction[1],
                    fullinteraction[2],
                    I,
                    F
                )
            )
            verbose && next!(p)
        end
    elseif threading == :multi
        for i in 1:Threads.nthreads()
            push!(fullrankblocks_perthread, MBF[])
            push!(nonzeros_perthread, 0)
        end

        Threads.@threads for fullinteraction in fullinteractions
            nonzeros_perthread[Threads.threadid()] += 
                length(fullinteraction[1].data)*length(fullinteraction[2].data)
            push!(
                fullrankblocks_perthread[Threads.threadid()],
                getfullmatrixview(
                    matrixassembler,
                    fullinteraction[1],
                    fullinteraction[2],
                    I,
                    F
                )
            )
            verbose && next!(p)
        end

        for i in eachindex(fullrankblocks_perthread)
            append!(fullrankblocks, fullrankblocks_perthread[i])
        end
    end

    MBL = MatrixBlock{I, F, LowRankMatrix{F}}
    lowrankblocks_perthread = Vector{MBL}[]
    lowrankblocks = MBL[]

    if verbose
        p = Progress(length(compressableinteractions), desc="Compressing far interactions: ")
    end

    if threading == :single
        for compressableinteraction in compressableinteractions
            push!(
                lowrankblocks, 
                getcompressedmatrix(
                    farmatrixassembler,
                    compressableinteraction[1],
                    compressableinteraction[2],
                    I,
                    F,
                    compressor=compressor,
                    tol=tol,
                    maxrank=maxrank,
                    svdrecompress=svdrecompress
                )
            )
            nonzeros += nnz(lowrankblocks[end])
            verbose && next!(p)
        end
    elseif threading == :multi
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
                    F,
                    compressor=compressor,
                    tol=tol,
                    maxrank=maxrank,
                    svdrecompress=svdrecompress
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

    return HMatrix{I, F}(
        fullrankblocks,
        lowrankblocks,
        rowdim,
        coldim,
        nonzeros + sum(nonzeros_perthread),
        maxrank,
        threading == :multi ?  true : false
    )
end

function computerinteractions!(
    testnode,#::BoxTreeNode,
    sourcenode,#::BoxTreeNode,
    fullinteractions,#::Vector{SVector{2,BoxTreeNode}},
    compressableinteractions#::Vector{SVector{2,BoxTreeNode}})
)
    if iscompressable(sourcenode, testnode)
        if sourcenode.level == 0 && testnode.level == 0
            push!(compressableinteractions, SVector(testnode, sourcenode))
            return
        else
            error("We do not expect this behavior")
        end
    end

    if sourcenode.children === nothing && testnode.children === nothing
        push!(fullinteractions, SVector(testnode, sourcenode))
        return
    else
        if sourcenode.children === nothing
            schild = sourcenode
            for tchild in testnode.children
                decide_compression(
                    tchild, 
                    schild, 
                    fullinteractions, 
                    compressableinteractions
                )
            end
        elseif testnode.children === nothing
            tchild = testnode
            for schild in sourcenode.children
                decide_compression(
                    tchild, 
                    schild, 
                    fullinteractions, 
                    compressableinteractions
                )
            end
        else
            for schild in sourcenode.children
                for tchild in testnode.children
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
    if length(ttchild.data) > 0 && length(sschild.data) > 0
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

function iscompressable(sourcenode::BoxTreeNode, testnode::BoxTreeNode)
    mindistance = 
        sqrt(length(sourcenode.boundingbox.center))*sourcenode.boundingbox.halflength 
    mindistance += 
        sqrt(length(testnode.boundingbox.center))*testnode.boundingbox.halflength 

    distance_centers = norm(sourcenode.boundingbox.center - testnode.boundingbox.center)
    if distance_centers > 1.1*mindistance
        return true
    else 
        return false
    end
end

function getfullmatrixview(
    matrixassembler,
    testnode,
    sourcenode,
    ::Type{I},
    ::Type{F};
) where {I, F}
    matrix = zeros(F, length(testnode.data), length(sourcenode.data))
    matrixassembler(matrix, testnode.data, sourcenode.data)

    return MatrixBlock{I, F, Matrix{F}}(
        matrix,
        testnode.data,
        sourcenode.data
    )
end

function getcompressedmatrix(
    matrixassembler::Function,
    testnode,
    sourcenode,
    ::Type{I},
    ::Type{F};
    tol=1e-4,
    maxrank=100,
    compressor=:aca,
    svdrecompress=true,
) where {I, F}

        #U, V = aca_compression2(assembler, kernel, testpoints, sourcepoints, 
        #testnode.data, sourcenode.data; tol=tol)
        #println("Confirm level: ", sourcenode.level)
        #print("a")
        #println(typeof(testnode))
        U, V = aca_compression(
            matrixassembler, 
            testnode.data, 
            sourcenode.data,
            F; 
            tol=tol,
            maxrank=maxrank,
            svdrecompress=svdrecompress
        )

        lm = MatrixBlock{I, F, LowRankMatrix{F}}(
            LowRankMatrix(U, V),
            testnode.data,
            sourcenode.data
        )

    return lm
end