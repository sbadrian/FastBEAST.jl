using FastBEAST
using StaticArrays
using LinearAlgebra
using CompScienceMeshes
using Printf
CM = CompScienceMeshes

## Example1 Evenly distributed sphere
sphere = CM.meshsphere(1,0.05)
spoints = sphere.vertices
spoints = spoints[2:length(spoints)] # First point is [0,0,0]
npoints = length(spoints)

## Example2 Evenly distributed cube
x = 1
y = 1
z = 1
h = 0.1
cuboid = CM.meshcuboid(x, y, z,h)
spoints = cuboid.vertices
npoints = length(spoints)

## Example2 Evenly distributed cylinder
radius = 1.0
height = 2.0
h = 0.1
cylinder = CM.meshcylinder(;radius, height, h)
spoints = cylinder.vertices
spoints = spoints[2:length(spoints)]
npoints = length(spoints)

##
@views OneoverRkernelassembler(matrix, tdata, sdata) = assembler(
    OneoverRkernel,
    matrix,
    spoints[tdata],
    spoints[sdata]
)

stree = create_tree(spoints, BoxTreeOptions(nmin=50))
kmat = assembler(OneoverRkernel, spoints, spoints)
@time hmat = HMatrix(OneoverRkernelassembler, stree, stree, compressor=:aca, Int64, Float64)

@printf("Accuracy test: %.2e\n", estimate_reldifference(hmat,kmat))
@printf("Compression rate: %.2f %%\n", compressionrate(hmat)*100)

stree = create_tree(spoints, KMeansTreeOptions(iterations=100,nchildren=2,nmin=50))
kmat = assembler(OneoverRkernel, spoints, spoints)
@time hmat = HMatrix(OneoverRkernelassembler, stree, stree, compressor=:aca, Int64, Float64)

@printf("Accuracy test: %.2e\n", estimate_reldifference(hmat,kmat))
@printf("Compression rate: %.2f %%\n", compressionrate(hmat)*100)

## Plot Figure
scatter(
    [spoints[i][1] for i=1:npoints], 
    [spoints[i][2] for i=1:npoints],
    [spoints[i][3] for i=1:npoints]
)

