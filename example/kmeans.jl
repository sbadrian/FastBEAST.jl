using FastBEAST
using Plots
using StaticArrays
plotlyjs()

## Example1 random distribution
points2D = [@SVector rand(2) for i=1:1000] + [SVector(1.0, 2.0) for i=1:1000]
scatter(
    [points2D[i][1] for i=1:length(points2D)], 
    [points2D[i][2] for i=1:length(points2D)]
)

## Example1 a) KMeans with single iteration
@time tree = create_tree(
    points2D,
    KMeansTreeOptions(iterations=1,nchildren=2,nmin=10)
)

scatter(
    [points2D[tree.children[1].data[i]][1] for i=1:length(tree.children[1].data)], 
    [points2D[tree.children[1].data[i]][2] for i=1:length(tree.children[1].data)]
)
scatter!(
    [points2D[tree.children[2].data[i]][1] for i=1:length(tree.children[2].data)], 
    [points2D[tree.children[2].data[i]][2] for i=1:length(tree.children[2].data)]
)

## Example1 b) KMeans with 10 iterations
@time tree = create_tree(
    points2D,
    KMeansTreeOptions(iterations=10,nchildren=2,nmin=10)
)

scatter(
    [points2D[tree.children[1].data[i]][1] for i=1:length(tree.children[1].data)], 
    [points2D[tree.children[1].data[i]][2] for i=1:length(tree.children[1].data)]
)
scatter!(
    [points2D[tree.children[2].data[i]][1] for i=1:length(tree.children[2].data)], 
    [points2D[tree.children[2].data[i]][2] for i=1:length(tree.children[2].data)]
)

## Example1 c) KMeans with quadtree, single iteration
nchildren=4
@time tree = create_tree(
    points2D, 
    KMeansTreeOptions(iterations=1,nchildren=nchildren,nmin=10)
)

scatter(
    [points2D[tree.children[1].data[i]][1] for i=1:length(tree.children[1].data)], 
    [points2D[tree.children[1].data[i]][2] for i=1:length(tree.children[1].data)]
)
for j = 2:nchildren
    scatter!(
        [points2D[tree.children[j].data[i]][1] for i=1:length(tree.children[j].data)], 
        [points2D[tree.children[j].data[i]][2] for i=1:length(tree.children[j].data)]
    )
end
scatter!()

## Example1 d) KMeans with quadtree, 10 iterations
nchildren=4
@time tree = create_tree(
    points2D, 
    KMeansTreeOptions(iterations=10,nchildren=nchildren,nmin=10)
);

scatter(
    [points2D[tree.children[1].data[i]][1] for i=1:length(tree.children[1].data)], 
    [points2D[tree.children[1].data[i]][2] for i=1:length(tree.children[1].data)]
)
for j = 2:nchildren
    scatter!(
        [points2D[tree.children[j].data[i]][1] for i=1:length(tree.children[j].data)], 
        [points2D[tree.children[j].data[i]][2] for i=1:length(tree.children[j].data)]
    )
end
scatter!()

## Example2 Larger problem
points3D = [@SVector rand(3) for i=1:100000] + [SVector(0.0, 0.0, 0.0) for i=1:100000]
@time tree = create_tree(
    points3D, 
    KMeansTreeOptions(iterations=20,nchildren=2,nmin=100)
);
