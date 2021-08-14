using LinearAlgebra

abstract type MatrixView{T} end

import Base.:eltype

function eltype(mv::MT) where MT <:MatrixView
    return typeof(mv).parameters[1]
end

struct Adjoint{T} 
    mv::T
end

import Base.:adjoint
function adjoint(mv::T) where T
    return Adjoint(mv)
end

function adjoint(adjmv::Adjoint{T}) where T
    return adjmv.mv
end

function eltype(mv::Adjoint{MT}) where MT <:MatrixView
    return typeof(adjoint(mv)).parameters[1]
end

struct FullMatrixView{T} <: MatrixView{T}
    matrix::Matrix{T}
    rightindices::Vector{Integer}
    leftindices::Vector{Integer}
    rowdim::Integer
    columndim::Integer
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



import Base.:*
function *(fmv::FMT, vecin::VT) where {FMT <:FullMatrixView, VT <: AbstractVector}
    T = promote_type(eltype(fmv), eltype(vecin))
    vecout = zeros(T, fmv.rowdim)

    vecout[fmv.leftindices] = fmv.matrix * vecin[fmv.rightindices]
    return vecout
end

function *(afmv::Adjoint{FMT}, vecin::VT) where {FMT <:FullMatrixView, VT <: AbstractVector}
    T = promote_type(eltype(afmv), eltype(vecin))
    vecout = zeros(T, afmv.mv.columndim)

    vecout[afmv.mv.rightindices] = adjoint(afmv.mv.matrix) * vecin[afmv.mv.leftindices]
    return vecout
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

function *(almv::Adjoint{LMT}, vecin::VT) where {LMT <: LowRankMatrixView, VT <: AbstractVector}
    T = promote_type(eltype(almv), eltype(vecin))
    vecout = zeros(T, almv.mv.columndim)

    vecout[almv.mv.rightindices] = almv.mv.rightmatrix*(almv.mv.leftmatrix' * vecin[almv.mv.leftindices])
    return vecout
end

function isapprox(hmat::AbstractHierarchicalMatrix, mat::AbstractMatrix)
    
end

abstract type AbstractHierarchicalMatrix{T} end

struct HMatrix{T} <: AbstractHierarchicalMatrix{T}
    matrixviews::Vector{MatrixView{T}}
    rowdim::Integer
    columndim::Integer
end

function adjoint(mv::HMatrix)
    return Adjoint(mv)
end

function eltype(hmat::HT) where HT <:AbstractHierarchicalMatrix
    return typeof(hmat).parameters[1]
end

function eltype(hmat::Adjoint{HT}) where HT <:AbstractHierarchicalMatrix
    return typeof(adjoint(hmat)).parameters[1]
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

function *(hmat::Adjoint{HT}, vecin::VT)  where {HT <: HMatrix, VT <: AbstractVector}
    if length(vecin) != hmat.mv.rowdim
        error("HMatrix vector and matrix have not matching dimensions")
    end

    T = promote_type(eltype(hmat), eltype(vecin))

    vecout = zeros(T, hmat.mv.columndim)

    for mv in hmat.mv.matrixviews
        vecout += adjoint(mv)*vecin
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


function adjoint(hmat::T) where T <: AbstractHierarchicalMatrix

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

    function build_interaction(asmpackage, matrixviews, schild, tchild)
        if length(schild.data) > 0 && length(tchild.data) > 0
            if iscompressable(schild, tchild)
                push!(matrixviews, getcompressedmatrix(asmpackage, schild, tchild))
            else
                hmatrixassembler!(asmpackage, matrixviews, schild, tchild)
            end
        end
    end

    if iscompressable(sourcenode, testnode)
        if sourcenode.level == 0 && testnode.level == 0
            push!(matrixviews, getcompressedmatrix(asmpackage, sourcenode, testnode))
            return
        else
            error("We do not expect this behavior")
        end
    end

    if sourcenode.children === nothing && testnode.children === nothing
        push!(matrixviews, getfullmatrixview(asmpackage, sourcenode, testnode))
    else
        if sourcenode.children === nothing
            schild = sourcenode
            for tchild in testnode.children
                build_interaction(asmpackage, matrixviews, schild, tchild)
            end
        elseif testnode.children === nothing
            tchild = testnode
            for schild in sourcenode.children
                build_interaction(asmpackage, matrixviews, schild, tchild)
            end
        else
            for schild in sourcenode.children
                for tchild in testnode.children
                    build_interaction(asmpackage, matrixviews, schild, tchild)
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