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

function nnz(fmv::FullMatrixView)
    return size(fmv.matrix,1)*size(fmv.matrix,2)
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
    vecout[lmv.leftindices] = lmv.leftmatrix*(lmv.rightmatrix * vecin[lmv.rightindices])
    return vecout
end

function *(almv::Adjoint{LMT}, vecin::VT) where {LMT <: LowRankMatrixView, VT <: AbstractVector}
    T = promote_type(eltype(almv), eltype(vecin))
    vecout = zeros(T, almv.mv.columndim)

    vecout[almv.mv.rightindices] = almv.mv.rightmatrix'*(almv.mv.leftmatrix' * vecin[almv.mv.leftindices])
    return vecout
end

function nnz(lmv::LowRankMatrixView)
    return size(lmv.rightmatrix,1)*size(lmv.rightmatrix,2) + 
            size(lmv.leftmatrix,1)*size(lmv.leftmatrix,2)
end

abstract type AbstractHierarchicalMatrix{T} end

struct HMatrix{T} <: AbstractHierarchicalMatrix{T}
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
#function adjoint(mv::HMatrix)
#    return Adjoint(mv)
#end

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

    for mv in hmat.matrixviews
        vecout += mv*vecin
    end

    return vecout
end

#function *(adjA::Adjoint{<:Any,<:AbstractMatrix{T}}, x::AbstractVector{S}) where {T,S}
#    TS = promote_op(matprod, T, S)
#    mul!(similar(x, TS, size(adjA, 1)), adjA, x)
#end

#function *(hmat::Adjoint{<:Any,<:AbstractHierarchicalMatrix{K}}, vecin::AbstractVector{S})  where {K,S}
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

function estimate_norm(mat; tol=1e-4)
    v = rand(size(mat,2))

    v = norm(v)
    itermin = 3
    i = 1
    σold = 1
    σnew = 1
    while norm(sqrt(σold)-sqrt(σnew))/norm(sqrt(σold)) > tol || i < itermin
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
                 tol=1e-4)
    
    T = Float64 #promote_type(eltype(sourcepoints[1]), eltype(testpoints[1]))

    #estimated_nlevel = maximum(log(8,length(testtree.data)), log(8,length(sourcetree.data)))
    #estimate_ninteractions = 
    maxsources = max(length(testtree.data), length(sourcetree.data))
    matrixviews = Vector{LowRankMatrixView{T}}(undef, 2*maxsources)

    rowdim = length(testtree.data)
    coldim = length(sourcetree.data)

    nlrmv = 0

    @time nonzeros, nlrmv = hmatrixassembler!(matrixassembler, 
                        matrixviews,
                        testtree,
                        sourcetree,
                        rowdim,
                        coldim,
                        nlrmv,
                        compressor=compressor,
                        tol=tol,
                        isdebug=isdebug)

    return HMatrix{T}(matrixviews, 
                      rowdim,
                      coldim,
                      nonzeros)
end

function hmatrixassembler!(matrixassembler::Function,
    matrixviews::Vector{LowRankMatrixView{Float64}},
    testnode::BoxTreeNode,
    sourcenode::BoxTreeNode,
    rowdim::Integer,
    coldim::Integer,
    nlrmv::Integer;
    compressor=:aca,
    tol=1e-4,
    isdebug=false)

    nonzeros = 0

    if iscompressable(sourcenode, testnode)
        if sourcenode.level == 0 && testnode.level == 0
            nlrmv += 1
            #matrixviews[nlrmv] = 
            getcompressedmatrix(matrixassembler,
                                        testnode,
                                        sourcenode,
                                        rowdim,
                                        coldim,
                                        compressor=compressor,
                                        tol=tol,
                                        isdebug=isdebug)

            #nonzeros += nnz(matrixviews[nlrmv])
            return nonzeros, nlrmv
        else
            error("We do not expect this behavior")
        end
    end

    if sourcenode.children === nothing && testnode.children === nothing
        nmv = getfullmatrixview(matrixassembler, testnode, sourcenode, rowdim, coldim)
        #push!(matrixviews, nmv)
        nonzeros += nnz(nmv)
    else
        if sourcenode.children === nothing
            schild = sourcenode
            for tchild in testnode.children
                newnonzeros, nlrmv = build_interaction(matrixassembler, 
                                                        matrixviews, 
                                                        tchild, 
                                                        schild, 
                                                        rowdim, 
                                                        coldim, 
                                                        nlrmv,  
                                                        tol=tol, 
                                                        compressor=compressor, 
                                                        isdebug=isdebug)
                nonzeros += newnonzeros
            end
        elseif testnode.children === nothing
            tchild = testnode
            for schild in sourcenode.children
                newnonzeros, nlrmv = build_interaction(matrixassembler, 
                                                        matrixviews, 
                                                        tchild, 
                                                        schild, 
                                                        rowdim, 
                                                        coldim, 
                                                        nlrmv,  
                                                        tol=tol, 
                                                        compressor=compressor, 
                                                        isdebug=isdebug)
                nonzeros += newnonzeros
            end
        else
            for schild in sourcenode.children
                for tchild in testnode.children
                    newnonzeros, nlrmv = build_interaction(matrixassembler, 
                                                            matrixviews, 
                                                            tchild, 
                                                            schild, 
                                                            rowdim, 
                                                            coldim, 
                                                            nlrmv,  
                                                            tol=tol, 
                                                            compressor=compressor, 
                                                            isdebug=isdebug)
                    nonzeros += newnonzeros                
                end
            end
        end
    end
    
    return nonzeros, nlrmv
    #if iswellseparated(sourcenode, testnode)
    #
    #end
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

function getfullmatrixview(matrixassembler, testnode, sourcenode, rowdim, coldim)
    matrix = zeros(Float64, length(testnode.data), length(sourcenode.data))
    matrixassembler(matrix, testnode.data, sourcenode.data)
    return FullMatrixView(matrix,
    sourcenode.data,
    testnode.data,
    rowdim,
    coldim)
end

function getcompressedmatrix(matrixassembler::Function, testnode, sourcenode, rowdim, coldim; 
                            tol = 1e-4, compressor=:aca, isdebug=false)

        #U, V = aca_compression2(assembler, kernel, testpoints, sourcepoints, 
        #testnode.data, sourcenode.data; tol=tol)
        #println("Confirm level: ", sourcenode.level)
        U, V = aca_compression(matrixassembler, testnode, sourcenode; tol=tol, isdebug=isdebug)

        lm::LowRankMatrixView{Float64} = LowRankMatrixView(V,
                                 U,
                                 sourcenode.data,
                                 testnode.data,
                                 rowdim,
                                 coldim)

    return lm
end

function build_interaction(matrixassembler, matrixviews, ttchild, sschild, rowdim, coldim, nlrmv::Int64; tol=1e-4, compressor=:aca, isdebug=false)
    if length(sschild.data) > 0 && length(ttchild.data) > 0
        if iscompressable(sschild, ttchild)
            nlrmv += 1
            #println(nlrmv)
            matrixviews[nlrmv] = 
            getcompressedmatrix(matrixassembler,
                                        ttchild,
                                        sschild,
                                        rowdim,
                                        coldim,
                                        tol=tol,
                                        compressor=compressor,
                                        isdebug=isdebug)
            #push!(matrixviews, nmv)
            newnonzeros::Int64 = nnz(matrixviews[nlrmv])
            #newnonzeros = 1
            return newnonzeros, nlrmv
            #return 1, 1
        else
            return hmatrixassembler!(matrixassembler,
                                            matrixviews,
                                            ttchild,
                                            sschild,
                                            rowdim,
                                            coldim,
                                            nlrmv,
                                            compressor=compressor,
                                            tol=tol,
                                            isdebug=isdebug)
        end
    end
end