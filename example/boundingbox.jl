using StaticArrays
using FastBEAST
using Plots
plotlyjs()

points2D = [@SVector rand(2) for i=1:100] + [SVector(1.0, 2.0) for i=1:100]

bbox2D = BoundingBox(points2D)
bboxframe2D = getboxframe(bbox2D)

scatter([points2D[i][1] for i=1:length(points2D)], 
        [points2D[i][2] for i=1:length(points2D)])
plot!(bboxframe2D[:,1], bboxframe2D[:,2])

##
points3D = [@SVector rand(3) for i=1:100] + [SVector(0.0, 0.0, 0.0) for i=1:100]

bbox3D = BoundingBox(points3D)
bboxframe3D = getboxframe(bbox3D)

scatter([points3D[i][1] for i=1:length(points3D)], 
        [points3D[i][2] for i=1:length(points3D)], 
        [points3D[i][3] for i=1:length(points3D)],
        markersize = 2)

plot!(  bboxframe3D[:,1], bboxframe3D[:,2], bboxframe3D[:,3], 
        line=(4, :black, :solid), 
        label = nothing)
