using LinearAlgebra

abstract type MatrixView{T} end

struct FullMatrixView{T} <: MatrixView{T}
    matrix::Matrix{T}
    rightindices::Vector{Integer}
    leftindices::Vector{Integer}
    rowdim::Integer
    columndim::Integer
end

import Base.:eltype

function eltype(mv::MT) where MT <:MatrixView
    return typeof(mv).parameters[1]
end

import Base.:*
function *(fmv::FMT, vecin::VT) where {FMT <:FullMatrixView, VT <: AbstractVector}
    T = promote_type(eltype(fmv), eltype(vecin))
    vecout = zeros(T, fmv.rowdim)

    vecout[fmv.leftindices] = fmv.matrix * vecin[fmv.rightindices]
    return vecout
end

function FullMatrixView(matrix::Matrix{T}, 
                        rightindices::Vector{I}, 
                        leftindices::Vector{I},
                        rowdim::Integer,
                        columndim::Integer) where {T, I <: Integer}

    return FullMatrixView{T}(matrix, 
                             rightindices, 
                             leftindices, 
                             rowdim, 
                             columndim)
end

struct LowRankMatrixView{T} <: MatrixView{T}
    rightmatrix::Matrix{T}
    leftmatrix::Matrix{T}
    rightindices::Vector{Integer}
    leftindices::Vector{Integer}
    rowdim::Integer
    columndim::Integer
end

function LowRankMatrixView(rightmatrix::Matrix{T}, 
                            leftmatrix::Matrix{T}, 
                            rightindices::Vector{I},
                            leftindices::Vector{I},
                            rowdim::Integer,
                            columndim::Integer) where {T, I <: Integer}

    return LowRankMatrixView{T}(rightmatrix,
                                leftmatrix,
                                rightindices, 
                                leftindices, 
                                rowdim, 
                                columndim)
end

function *(lmv::LMT, vecin::VT) where {LMT <:LowRankMatrixView, VT <: AbstractVector}
    T = promote_type(eltype(lmv), eltype(vecin))
    vecout = zeros(T, lmv.rowdim)

    vecout[lmv.leftindices] = lmv.leftmatrix*(lmv.rightmatrix' * vecin[lmv.rightindices])
    return vecout
end

abstract type AbstractHierarchicalMatrix{T} end

struct HMatrix{T} <: AbstractHierarchicalMatrix{T}
    matrixviews::Vector{MatrixView{T}}
    rowdim::Integer
    columndim::Integer
end

function eltype(hmatrix::T) where {T <: AbstractHierarchicalMatrix}
    return typeof(hmatrix).parameters[1]
end

function *(hmat::HT, vecin::VT)  where {HT <: HMatrix, VT <: AbstractVector}
    if length(vecin) != hmat.columndim
        error("HMatrix vector and matrix have not matching dimensions")
    end

    T = promote_type(eltype(hmat), eltype(vecin))

    vecout = zeros(T, hmat.rowdim)

    for mv in hmat.matrixviews
        vecout += mv*vecin
    end

    return vecout
end

function HMatrix(asmpackage::Tuple, 
                 sourcetree::BoxTreeNode, 
                 testtree::BoxTreeNode)
    
    assembler, kernel, sourcepoints, testpoints = asmpackage

    T = promote_type(eltype(sourcepoints[1]), eltype(testpoints[1]))

    matrixviews = MatrixView{T}[]

    hmatrixassembler!(asmpackage, 
                        matrixviews,
                        sourcetree, 
                        testtree)

    return HMatrix{T}(matrixviews, 
                    length(testpoints),
                    length(sourcepoints))
end

#function * (hmatrix::HT, matrix::MT) where {HT <:AbstractHierarchicalMatrix, MT <: AbstractMatrix}
#
#end

function hmatrixassembler!(asmpackage::Tuple, 
    matrixviews::Vector{MT},
    sourcenode::BoxTreeNode, 
    testnode::BoxTreeNode) where MT <: MatrixView


    function getfullmatrixview(asmpackage, sourcenode, testnode)
        assembler, kernel, sourcepoints, testpoints = asmpackage

        return FullMatrixView(assembler(kernel, sourcepoints[sourcenode.data], testpoints[testnode.data]),
        sourcenode.data,
        testnode.data,
        length(testpoints),
        length(sourcepoints))
    end

    function getcompressedmatrix(asmpackage, sourcenode, testnode; tol = 1e-4)
        assembler, kernel, sourcepoints, testpoints = asmpackage

        fullmat = assembler(kernel, sourcepoints[sourcenode.data], testpoints[testnode.data])

        U,S,V = svd(fullmat)
        k = 1
        while k < length(S) && S[k] > S[1]*tol
            k += 1
        end

        return LowRankMatrixView(V[:, 1:k],
        U[:, 1:k]*diagm(S[1:k]),
        sourcenode.data,
        testnode.data,
        length(testpoints),
        length(sourcepoints))
    end

    if sourcenode.children === nothing && testnode.children === nothing
        push!(matrixviews, getfullmatrixview(asmpackage, sourcenode, testnode))
    elseif sourcenode.children !== nothing || testnode.children !== nothing
        if sourcenode.children === nothing
            for tchild in testnode.children
                push!(matrixviews, getfullmatrixview(asmpackage, sourcenode, tchild))
            end
        elseif testnode.children === nothing
            for schild in sourcenode.children
                push!(matrixviews, getfullmatrixview(asmpackage, schild, testnode))
            end
        else
            for schild in sourcenode.children
                for tchild in testnode.children
                    if length(schild.data) > 0 && length(tchild.data) > 0
                        if iscompressable(schild, tchild)
                            push!(matrixviews, getcompressedmatrix(asmpackage, schild, tchild))
                        else
                            hmatrixassembler!(asmpackage, matrixviews, schild, tchild)
                        end
                    end
                end
            end
        end
    end
    
    #if iswellseparated(sourcenode, testnode)
    #
    #end
end


function iscompressable(sourcenode::BoxTreeNode, testnode::BoxTreeNode)
    mindistance = sqrt(length(sourcenode.boundingbox.center))*sourcenode.boundingbox.halflength 
    mindistance += sqrt(length(testnode.boundingbox.center))*testnode.boundingbox.halflength 

    distance_centers = norm(sourcenode.boundingbox.center - testnode.boundingbox.center)
    if distance_centers > mindistance
        return true
    else 
        return false
    end
end