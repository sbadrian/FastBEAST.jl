using LinearAlgebra
using ProgressMeter

abstract type AbstractHierarchicalMatrix{T} end

struct HMatrix{T} <: AbstractHierarchicalMatrix{T}
    fullmatrixviews::Vector{FullMatrixView{T}}
    matrixviews::Vector{LowRankMatrixView{T}}
    rowdim::Integer
    columndim::Integer
    nnz::Integer
end

function nnz(hmat::HT) where HT <:AbstractHierarchicalMatrix
    return hmat.nnz
end

function compressionrate(hmat::HT) where HT <:AbstractHierarchicalMatrix
    fullsize = hmat.rowdim*hmat.columndim
    return (fullsize - nnz(hmat))/fullsize
end

function eltype(hmat::HT) where HT <:AbstractHierarchicalMatrix
    return typeof(hmat).parameters[1]
end

function eltype(hmat::Adjoint{HT}) where HT <:AbstractHierarchicalMatrix
    return typeof(adjoint(hmat)).parameters[1]
end

import Base.:size

function size(hmat::AbstractHierarchicalMatrix, dim=nothing)
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

function size(hmat::Adjoint{T}, dim=nothing) where T <: AbstractHierarchicalMatrix
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

function *(hmat::HT, vecin::VT)  where {HT <: HMatrix, VT <: AbstractVector}

    if length(vecin) != hmat.columndim
        error("HMatrix vector and matrix have not matching dimensions")
    end

    T = promote_type(eltype(hmat), eltype(vecin))

    vecout = zeros(T, hmat.rowdim)

    for fmv in hmat.fullmatrixviews
        vecout[fmv.leftindices] += fmv.matrix * vecin[fmv.rightindices]
    end

    for lmv in hmat.matrixviews
        vecout[lmv.leftindices] += lmv.leftmatrix*(lmv.rightmatrix * vecin[lmv.rightindices])
    end

    return vecout
end

function *(hmat::Adjoint{HT}, vecin::VT)  where {HT <: HMatrix, VT <: AbstractVector}
    if length(vecin) != hmat.mv.rowdim
        error("HMatrix vector and matrix have not matching dimensions")
    end

    T = promote_type(eltype(hmat), eltype(vecin))

    vecout = zeros(T, hmat.mv.columndim)

    for afmv in hmat.mv.fullmatrixviews
        vecout[afmv.rightindices] += adjoint(afmv.matrix) * vecin[afmv.leftindices]
    end

    for almv in hmat.mv.matrixviews
        vecout[almv.rightindices] += almv.rightmatrix'*(almv.leftmatrix' * vecin[almv.leftindices])
    end

    return vecout
end

function estimate_norm(mat; tol=1e-4, itmax = 1000)
    v = rand(size(mat,2))

    v = v/norm(v)
    itermin = 3
    i = 1
    σold = 1
    σnew = 1
    while (norm(sqrt(σold)-sqrt(σnew))/norm(sqrt(σold)) > tol || i < itermin) && i < itmax
        σold = σnew
        w = mat*v
        x = adjoint(mat)*w
        σnew = norm(x)
        v = x/norm(x)
        i += 1
    end
    return sqrt(σnew)
end

function estimate_reldifference(hmat, refmat; tol=1e-4)
    #if size(hmat) != size(refmat)
    #    error("Dimensions of matrices do not match")
    #end
    
    v = rand(size(hmat,2))

    v = v/norm(v)
    itermin = 3
    i = 1
    σold = 1
    σnew = 1
    while norm(sqrt(σold)-sqrt(σnew))/norm(sqrt(σold)) > tol || i < itermin
        σold = σnew
        w = hmat*v - refmat*v
        x = adjoint(hmat)*w - adjoint(refmat)*w
        σnew = norm(x)
        v = x/norm(x)
        i += 1
    end

    norm_refmat = estimate_norm(refmat, tol=tol)

    return sqrt(σnew)/norm_refmat
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

    fullmatrixviews = FullMatrixView{T, I}[]
    matrixviews = LowRankMatrixView{T, I}[]

    rowdim = length(testtree.data)
    coldim = length(sourcetree.data)

    nonzeros = 0
    println("Number of full interactions: ", length(fullinteractions))
    println("Number of compressable interactions: ", length(compressableinteractions))

    @showprogress "Computing full interactions: " for fullinteraction in fullinteractions
        nonzeros += length(fullinteraction[1].data)*length(fullinteraction[2].data)
        push!(fullmatrixviews, getfullmatrixview(matrixassembler, 
                                                fullinteraction[1], 
                                                fullinteraction[2], 
                                                rowdim, 
                                                coldim, 
                                                T=T))
    end

    @showprogress "Compressing far interactions: " for compressableinteraction in compressableinteractions
        push!(matrixviews, getcompressedmatrix(matrixassembler,
                                                compressableinteraction[1],
                                                compressableinteraction[2],
                                                rowdim,
                                                coldim,
                                                compressor=compressor,
                                                tol=tol,
                                                T=T))
        nonzeros += nnz(matrixviews[end])

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