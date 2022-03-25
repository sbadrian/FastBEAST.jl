using Test
using LinearAlgebra
using FastBEAST
using StaticArrays

points2D = [SVector(0.1, 0.1), #1
            SVector(1.0, 1.0), #2
            SVector(0.2, 0.2), #3
            SVector(1.1, 1.1)] #4

root = create_tree(points2D, KMeansTreeOptions(algorithm=:naive))

@test length(root.children) == 2
@test FastBEAST.indices(root.children[1])[1] == 1
@test FastBEAST.indices(root.children[1])[2] == 3
@test FastBEAST.indices(root.children[2])[1] == 2
@test FastBEAST.indices(root.children[2])[2] == 4
@test root.children[1].radius ≈ norm([0.05 0.05]) atol=1e-15