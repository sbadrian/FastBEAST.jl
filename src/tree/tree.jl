abstract type TreeOptions end

## Box tree
# Implements quad- (2D) and oct-tree (3D) cluster strategies
include("boundingbox.jl")
include("boxtree.jl")

## KMeans-Tree
include("kmeans.jl")

## Default functions
function create_tree(points::Vector{SVector{D,T}})
    create_function(points, BoxTreeOptions())
end