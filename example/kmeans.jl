using FastBEAST
using Plots
using StaticArrays
using BenchmarkTools
plotlyjs()

## Example1 Visualisation of KMeans clusters of random distribution.
points2D = [@SVector rand(2) for i=1:1000] + [SVector(1.0, 2.0) for i=1:1000]

tree = create_tree(
    points2D,
    KMeansTreeOptions()
)

scatter(
    [points2D[tree.children[1].data.indices[i]][1] for i=1:length(tree.children[1].data.indices)], 
    [points2D[tree.children[1].data.indices[i]][2] for i=1:length(tree.children[1].data.indices)]
)
scatter!(
    [points2D[tree.children[2].data.indices[i]][1] for i=1:length(tree.children[2].data.indices)], 
    [points2D[tree.children[2].data.indices[i]][2] for i=1:length(tree.children[2].data.indices)]
)


## Example2 Comparison of BoxTree and different KMeans approaches
points = [@SVector rand(3) for i=1:100000] + [SVector(0.0, 0.0, 0.0) for i=1:100000]

@btime tree = create_tree(
    points, 
    KMeansTreeOptions(iterations=20,nchildren=2, nmin=100)
);

@btime tree = create_tree(
    points, 
    KMeansTreeOptions(iterations=20, nchildren=2, nmin=100, algorithm=:naive)
);

@btime tree = create_tree(
    points, 
    BoxTreeOptions(nmin=100)
);

## Example3 
points = [@SVector rand(3) for i=1:1000000] + [SVector(0.0, 0.0, 0.0) for i=1:1000000]

@time tree = create_tree(
    points, 
    KMeansTreeOptions(iterations=20,nchildren=2, nmin=100)
);

@time tree = create_tree(
    points, 
    BoxTreeOptions(nmin=100)
);