module FastBEAST
using LinearAlgebra
include("tree/tree.jl")

include("aca/aca_utils.jl")
include("aca/pivoting.jl")
include("aca/convergence.jl")
include("aca/aca.jl")
include("skeletons.jl")
include("hmatrix.jl")
include("utils.jl")
include("fmm.jl")
include("beast.jl")
include("fmm/operators/FMMoperator.jl")

export BoundingBox
export getboxframe
export getchildbox
export whichchildbox

export BoxTreeNode
export create_tree
export BoxTreeOptions
export ExaFMMOptions
export KMeansTreeOptions
export KMeansTreeNode

export aca, allocate_aca_memory
export LazyMatrix

export MatrixBlock, LowRankMatrix

export HMatrix
export buildhmatrix
export adjoint
export estimate_norm
export estimate_reldifference
export nnz
export compressionrate

export hassemble
export fmmassemble
end
