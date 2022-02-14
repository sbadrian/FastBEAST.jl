abstract type TreeOptions end
abstract type AbstractNode{I, F, N} end

children(node::AbstractNode{I, F, N}) where {I, F, N} = node.children
numchildren(node::AbstractNode{I, F, N}) where {I, F, N} = length(node.children)
lastchild(node::AbstractNode{I, F, N}) where {I, F, N} = last(node.children)
level(node::AbstractNode{I, F, N}) where {I, F, N} = node.level

abstract type NodeData{I, F} end

indices(node::AbstractNode{I, F, N}) where {I, F, N <: NodeData{I, F}} = node.data.indices
numindices(node::AbstractNode{I, F, N}) where {I, F, N <: NodeData{I, F}} = length(node.data.indices)

struct ACAData{I, F} <: NodeData{I, F}
    indices::Vector{I}
end

function ACAData(indices::Vector{I}, ::Type{F}) where {I, F}
    return ACAData{I, F}(indices)
end


## Box tree
# Implements quad- (2D) and oct-tree (3D) cluster strategies
include("boundingbox.jl")
include("boxtree.jl")

## KMeans-Tree
include("kmeans.jl")

## Default functions
function create_tree(points::Vector{SVector{D,T}}) where {D,T}
    create_tree(points, BoxTreeOptions())
end