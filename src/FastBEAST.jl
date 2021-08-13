module FastBEAST

include("tree/boundingbox.jl")
include("tree/basetree.jl")

export BoundingBox
export getboxframe
export getchildbox
export whichchildbox

export BoxTreeNode
export create_tree

end
