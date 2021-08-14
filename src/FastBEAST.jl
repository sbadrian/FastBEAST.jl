module FastBEAST

include("tree/boundingbox.jl")
include("tree/basetree.jl")
include("hmatrix.jl")

export BoundingBox
export getboxframe
export getchildbox
export whichchildbox

export BoxTreeNode
export create_tree

export HMatrix
export buildhmatrix
export adjoint

end
