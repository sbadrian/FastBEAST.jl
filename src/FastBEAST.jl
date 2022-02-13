module FastBEAST
using LinearAlgebra
include("tree/tree.jl")

include("aca.jl")
include("skeletons.jl")

export BoundingBox
export getboxframe
export getchildbox
export whichchildbox

export BoxTreeNode
export create_tree
export BoxTreeOptions
export KMeansTreeOptions
export KMeansTreeNode

export aca_compression

export BlockMatrix, LowRankMatrix, LowRankBlock

export HMatrix
export buildhmatrix
export adjoint
export estimate_norm
export estimate_reldifference
export nnz
export compressionrate

export hassemble
end
