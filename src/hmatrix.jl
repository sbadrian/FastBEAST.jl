using LinearAlgebra
using LinearMaps
using ProgressMeter

struct HMatrix{T,I} <: LinearMaps.LinearMap{T}
    fullmatrixviews::Vector{FullMatrixView{T,I}}
    matrixviews::Vector{LowRankMatrixView{T,I}}
    rowdim::I
    columndim::I
    nnz::I
    maxrank::I
end

function nnz(hmat::HMatrix) where HT <: HMatrix
    return hmat.nnz
end

function compressionrate(hmat::HT) where HT <: HMatrix
    fullsize = hmat.rowdim*hmat.columndim
    return (fullsize - nnz(hmat))/fullsize
end

import Base.:size

function size(hmat::HMatrix, dim=nothing)
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

function size(hmat::Adjoint{T}, dim=nothing) where T <: HMatrix
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

    b = zeros(eltype(y), 200)
    c = zeros(eltype(y), size(A,1))
    
    for fmv in A.fullmatrixviews
        mul!(c[1:size(fmv.matrix,1)], fmv.matrix, x[fmv.rightindices])
        y[fmv.leftindices] .+= c[1:size(fmv.matrix,1)]
        #y[fmv.leftindices] += fmv.matrix * x[fmv.rightindices]
    end
    
    for lmv in A.matrixviews
        mul!(b[1:size(lmv.rightmatrix,1)], lmv.rightmatrix, x[lmv.rightindices]) 
        mul!(c[1:size(lmv.leftmatrix,1)], lmv.leftmatrix, b[1:size(lmv.rightmatrix,1)])
        y[lmv.leftindices] .+= c[1:size(lmv.leftmatrix,1)]
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

    b = zeros(eltype(y), 200)
    c = zeros(eltype(y), size(transA,1))

    for afmv in transA.lmap.fullmatrixviews
        mul!(c[1:size(transpose(afmv.matrix),1)], transpose(afmv.matrix), x[afmv.leftindices])
        y[afmv.rightindices] .+= c[1:size(adjoint(afmv.matrix),1)]
        #y[afmv.rightindices] += adjoint(afmv.matrix) * x[afmv.leftindices]
    end

    for almv in transA.lmap.matrixviews
        mul!(b[1:size(almv.leftmatrix,2)], transpose(almv.leftmatrix), x[almv.leftindices]) 
        mul!(c[1:size(almv.rightmatrix,2)], transpose(almv.rightmatrix), b[1:size(almv.leftmatrix,2)])
        y[almv.rightindices] .+= c[1:size(almv.rightmatrix,2)]
        #y[almv.rightindices] += almv.rightmatrix'*(almv.leftmatrix' * x[almv.leftindices])
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

    b = zeros(eltype(y), 200)
    c = zeros(eltype(y), size(transA,1))

    for afmv in transA.lmap.fullmatrixviews
        mul!(c[1:size(adjoint(afmv.matrix),1)], adjoint(afmv.matrix), x[afmv.leftindices])
        y[afmv.rightindices] .+= c[1:size(adjoint(afmv.matrix),1)]
        #y[afmv.rightindices] += adjoint(afmv.matrix) * x[afmv.leftindices]
    end

    for almv in transA.lmap.matrixviews
        mul!(b[1:size(almv.leftmatrix,2)], almv.leftmatrix', x[almv.leftindices]) 
        mul!(c[1:size(almv.rightmatrix,2)], almv.rightmatrix', b[1:size(almv.leftmatrix,2)])
        y[almv.rightindices] .+= c[1:size(almv.rightmatrix,2)]
        #y[almv.rightindices] += almv.rightmatrix'*(almv.leftmatrix' * x[almv.leftindices])
    end

    return y
end

function HMatrix(matrixassembler::Function,
                 testtree::BoxTreeNode,
                 sourcetree::BoxTreeNode;
                 compressor=:naive,
                 tol=1e-4,
                 maxrank=100,
                 T=ComplexF64,
                 I=Int64,
                 threading=:single,
                 farmatrixassembler=matrixassembler,
                 verbose=false,
                 svdrecompress=true)
    
    fullinteractions = SVector{2,BoxTreeNode}[]
    compressableinteractions = SVector{2,BoxTreeNode}[]

    computerinteractions!(testtree,
                            sourcetree,
                            fullinteractions,
                            compressableinteractions)

    fullmatrixviews_perthread = Vector{FullMatrixView{T, I}}[]
    fullmatrixviews = FullMatrixView{T, I}[]

    rowdim = length(testtree.data)
    coldim = length(sourcetree.data)

    nonzeros_perthread = I[]
    nonzeros = 0
    verbose && println("Number of full interactions: ", length(fullinteractions))
    verbose && println("Number of compressable interactions: ", length(compressableinteractions))

    if verbose
        p = Progress(length(fullinteractions), desc="Computing full interactions: ")
    end

    if threading == :single
        for fullinteraction in fullinteractions
            nonzeros += length(fullinteraction[1].data)*length(fullinteraction[2].data)
            push!(fullmatrixviews, getfullmatrixview(matrixassembler,
                                                    fullinteraction[1],
                                                    fullinteraction[2],
                                                    rowdim,
                                                    coldim,
                                                    T=T, I=I))
            verbose && next!(p)
        end
    elseif threading == :multi
        for i in 1:Threads.nthreads()
            push!(fullmatrixviews_perthread, FullMatrixView{T, I}[])
            push!(nonzeros_perthread, 0)
        end

        Threads.@threads for fullinteraction in fullinteractions
            nonzeros_perthread[Threads.threadid()] += length(fullinteraction[1].data)*length(fullinteraction[2].data)
            push!(fullmatrixviews_perthread[Threads.threadid()], getfullmatrixview(matrixassembler,
                                                    fullinteraction[1],
                                                    fullinteraction[2],
                                                    rowdim,
                                                    coldim,
                                                    T=T, I=I))
            verbose && next!(p)
        end

        for i in eachindex(fullmatrixviews_perthread)
            append!(fullmatrixviews, fullmatrixviews_perthread[i])
        end
    end

    matrixviews_perthread = Vector{LowRankMatrixView{T, I}}[]
    matrixviews = LowRankMatrixView{T, I}[]

    if verbose
        p = Progress(length(compressableinteractions), desc="Compressing far interactions: ")
    end

    if threading == :single
        for compressableinteraction in compressableinteractions
            push!(matrixviews, getcompressedmatrix(farmatrixassembler,
                                                    compressableinteraction[1],
                                                    compressableinteraction[2],
                                                    rowdim,
                                                    coldim,
                                                    compressor=compressor,
                                                    tol=tol,
                                                    maxrank=maxrank,
                                                    svdrecompress=svdrecompress,
                                                    T=T, I=I))
            nonzeros += nnz(matrixviews[end])
            verbose && next!(p)
        end
    elseif threading == :multi
        for i in 1:Threads.nthreads()
            push!(matrixviews_perthread, LowRankMatrixView{T, I}[])
        end

        Threads.@threads for compressableinteraction in compressableinteractions
            push!(matrixviews_perthread[Threads.threadid()], getcompressedmatrix(farmatrixassembler,
                                                    compressableinteraction[1],
                                                    compressableinteraction[2],
                                                    rowdim,
                                                    coldim,
                                                    compressor=compressor,
                                                    tol=tol,
                                                    maxrank=maxrank,
                                                    svdrecompress=svdrecompress,
                                                    T=T, I=I))
            nonzeros_perthread[Threads.threadid()] += nnz(matrixviews_perthread[Threads.threadid()][end])
            verbose && next!(p)
        end

        for i in eachindex(matrixviews_perthread)
            append!(matrixviews, matrixviews_perthread[i])
        end
    end

    return HMatrix{T,I}(fullmatrixviews,
                      matrixviews, 
                      rowdim,
                      coldim,
                      nonzeros + sum(nonzeros_perthread),
                      maxrank)
end

function computerinteractions!(testnode::BoxTreeNode,
                                sourcenode::BoxTreeNode,
                                fullinteractions::Vector{SVector{2,BoxTreeNode}},
                                compressableinteractions::Vector{SVector{2,BoxTreeNode}})

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
                decide_compression(tchild, 
                                    schild, 
                                    fullinteractions, 
                                    compressableinteractions)
            end
        elseif testnode.children === nothing
            tchild = testnode
            for schild in sourcenode.children
                decide_compression(tchild, 
                                    schild, 
                                    fullinteractions, 
                                    compressableinteractions)
            end
        else
            for schild in sourcenode.children
                for tchild in testnode.children
                    decide_compression(tchild, 
                                        schild, 
                                        fullinteractions, 
                                        compressableinteractions)
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
            computerinteractions!(ttchild,
                                    sschild,
                                    fullinteractions,
                                    compressableinteractions)
        end
    end
end

function iscompressable(sourcenode::BoxTreeNode, testnode::BoxTreeNode)
    mindistance = sqrt(length(sourcenode.boundingbox.center))*sourcenode.boundingbox.halflength 
    mindistance += sqrt(length(testnode.boundingbox.center))*testnode.boundingbox.halflength 

    distance_centers = norm(sourcenode.boundingbox.center - testnode.boundingbox.center)
    if distance_centers > 1.1*mindistance
        return true
    else 
        return false
    end
end

function getfullmatrixview(matrixassembler, testnode, sourcenode, rowdim, coldim; T=ComplexF64, I=Int64)
    matrix = zeros(T, length(testnode.data), length(sourcenode.data))
    matrixassembler(matrix, testnode.data, sourcenode.data)
    return FullMatrixView{T,I}(matrix,
                            sourcenode.data,
                            testnode.data,
                            rowdim,
                            coldim)
end

function getcompressedmatrix(matrixassembler::Function, testnode, sourcenode, rowdim, coldim; 
                            tol = 1e-4, maxrank=100,  compressor=:aca, svdrecompress=true, 
                            T=ComplexF64, I=Int64)

        #U, V = aca_compression2(assembler, kernel, testpoints, sourcepoints, 
        #testnode.data, sourcenode.data; tol=tol)
        #println("Confirm level: ", sourcenode.level)
        U, V = aca_compression(matrixassembler, testnode, sourcenode; 
                                tol=tol, maxrank=maxrank, svdrecompress=svdrecompress, T=T)

        lm = LowRankMatrixView{T,I}(V,
                                 U,
                                 sourcenode.data,
                                 testnode.data,
                                 rowdim,
                                 coldim)

    return lm
end