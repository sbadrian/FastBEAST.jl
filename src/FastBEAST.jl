module FastBEAST

include("tree/tree.jl")

include("aca.jl")
include("skeletons.jl")
include("utils.jl")
include("hmatrix.jl")
include("beast.jl")

export BoundingBox
export getboxframe
export getchildbox
export whichchildbox

export BoxTreeNode
export create_tree

export aca_compression

export FullMatrixView, FullMatrixView2, MatrixView, LowRankMatrixView

export HMatrix
export buildhmatrix
export adjoint
export estimate_norm
export estimate_reldifference
export nnz
export compressionrate

export hassemble
end
