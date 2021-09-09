using LinearAlgebra
using LinearMaps
using ProgressMeter

struct HMatrix{T} <: LinearMaps.LinearMap{T}
    fullmatrixviews::Vector{FullMatrixView{T}}
    matrixviews::Vector{LowRankMatrixView{T}}
    rowdim::Integer
    columndim::Integer
    nnz::Integer
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

function LinearAlgebra.mul!(y::AbstractVecOrMat, A::HMatrix, x::AbstractVector)
    LinearMaps.check_dim_mul(y, A, x)

    fill!(y, zero(eltype(y)))

    for fmv in A.fullmatrixviews
        y[fmv.leftindices] += fmv.matrix * x[fmv.rightindices]
    end

    for lmv in A.matrixviews
        y[lmv.leftindices] += lmv.leftmatrix*(lmv.rightmatrix * x[lmv.rightindices])
    end

    return y
end


function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    transA::LinearMaps.TransposeMap{<:Any,<:HMatrix},
    x::AbstractVector
)
    LinearMaps.check_dim_mul(y, transA, x)

    fill!(y, zero(eltype(y)))

    for afmv in transA.lmap.fullmatrixviews
        y[afmv.rightindices] += adjoint(afmv.matrix) * x[afmv.leftindices]
    end

    for almv in transA.lmap.matrixviews
        y[almv.rightindices] += almv.rightmatrix'*(almv.leftmatrix' * x[almv.leftindices])
    end

    return y
end

function HMatrix(matrixassembler::Function,
                 testtree::BoxTreeNode,
                 sourcetree::BoxTreeNode;
                 compressor=:naive,
                 isdebug=false,
                 tol=1e-4,
                 T=ComplexF64,
                 I=Int64)
    
    fullinteractions = SVector{2,BoxTreeNode}[]
    compressableinteractions = SVector{2,BoxTreeNode}[]

    computerinteractions!(testtree,
                            sourcetree,
                            fullinteractions,
                            compressableinteractions)

    fullmatrixviews_perthread = Vector{FullMatrixView{T, I}}[]
    matrixviews = LowRankMatrixView{T, I}[]

    rowdim = length(testtree.data)
    coldim = length(sourcetree.data)

    nonzeros = 0
    println("Number of full interactions: ", length(fullinteractions))
    println("Number of compressable interactions: ", length(compressableinteractions))

    p = Progress(length(fullinteractions), desc="Computing full interactions: ")

    for i in 1:Threads.nthreads()
        push!(fullmatrixviews_perthread, FullMatrixView{T, I}[])
    end

    Threads.@threads for fullinteraction in fullinteractions
        nonzeros += length(fullinteraction[1].data)*length(fullinteraction[2].data)
        push!(fullmatrixviews_perthread[Threads.threadid()], getfullmatrixview(matrixassembler, 
                                                fullinteraction[1], 
                                                fullinteraction[2], 
                                                rowdim, 
                                                coldim, 
                                                T=T))
        next!(p)
    end

    fullmatrixviews = FullMatrixView{T, I}[]

    for i in 1:Threads.nthreads()
        append!(fullmatrixviews, fullmatrixviews_perthread[i])
    end

    p = Progress(length(fullinteractions), desc="Compressing far interactions: ")

    for compressableinteraction in compressableinteractions
        push!(matrixviews, getcompressedmatrix(matrixassembler,
                                                compressableinteraction[1],
                                                compressableinteraction[2],
                                                rowdim,
                                                coldim,
                                                compressor=compressor,
                                                tol=tol,
                                                T=T))
        nonzeros += nnz(matrixviews[end])
        next!(p)
    end

    return HMatrix{T}(fullmatrixviews,
                      matrixviews, 
                      rowdim,
                      coldim,
                      nonzeros)
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

function getfullmatrixview(matrixassembler, testnode, sourcenode, rowdim, coldim; T=ComplexF64)
    matrix = zeros(T, length(testnode.data), length(sourcenode.data))
    matrixassembler(matrix, testnode.data, sourcenode.data)
    return FullMatrixView(matrix,
                            sourcenode.data,
                            testnode.data,
                            rowdim,
                            coldim)
end

function getcompressedmatrix(matrixassembler::Function, testnode, sourcenode, rowdim, coldim; 
                            tol = 1e-4, compressor=:aca, T=ComplexF64)

        #U, V = aca_compression2(assembler, kernel, testpoints, sourcepoints, 
        #testnode.data, sourcenode.data; tol=tol)
        #println("Confirm level: ", sourcenode.level)
        U, V = aca_compression(matrixassembler, testnode, sourcenode; tol=tol, T=T)

        lm = LowRankMatrixView(V,
                                 U,
                                 sourcenode.data,
                                 testnode.data,
                                 rowdim,
                                 coldim)

    return lm
end