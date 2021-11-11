abstract type TreeOptions end

## Box tree
# Implements quad- (2D) and oct-tree (3D) cluster strategies
include("boundingbox.jl")
include("boxtree.jl") 
include("kmeans.jl")