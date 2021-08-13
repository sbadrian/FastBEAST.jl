
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

    matrixviews = FullMatrixView{T}[]

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

    assembler, kernel, sourcepoints, testpoints = asmpackage

    if sourcenode.children === nothing && testnode.children === nothing
        push!(matrixviews, FullMatrixView(assembler(kernel, sourcepoints[sourcenode.data], testpoints[testnode.data]),
                sourcenode.data,
                testnode.data,
                length(testpoints),
                length(sourcepoints)
                ))
    elseif sourcenode.children !== nothing || testnode.children !== nothing
        if sourcenode.children === nothing
            error("this case needs to be implement and test")
        elseif testnode.children === nothing
            error("this case needs to be implement and test")
        else
            for schild in sourcenode.children
                for tchild in testnode.children
                    hmatrixassembler!(asmpackage, matrixviews, schild, tchild)
                end
            end
        end
    end
    
    #if iswellseparated(sourcenode, testnode)
    #
    #end
end