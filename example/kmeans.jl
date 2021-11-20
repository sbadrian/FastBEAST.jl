using FastBEAST
using Plots
using StaticArrays
plotlyjs()

##
# Example1
points2D = [@SVector rand(2) for i=1:1000] + [SVector(1.0, 2.0) for i=1:1000]
scatter(
    [points2D[i][1] for i=1:length(points2D)], 
    [points2D[i][2] for i=1:length(points2D)]
)

##
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

##
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
##

# Example2
points2D = [@SVector rand(2) for i=1:1000] + [SVector(1.0, 2.0) for i=1:1000]
scatter(
    [points2D[i][1] for i=1:length(points2D)], 
    [points2D[i][2] for i=1:length(points2D)]
)

##
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

##
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
##

#Example3
points3D = [@SVector rand(3) for i=1:1000000] + [SVector(0.0, 0.0, 0.0) for i=1:1000000]
@time tree = create_tree(
    points3D, 
    KMeansTreeOptions(iterations=1,nchildren=2,nmin=100)
);
##