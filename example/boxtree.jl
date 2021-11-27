using FastBEAST
using Plots
using StaticArrays
plotlyjs()

##
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


##
points2D = [@SVector rand(2) for i=1:100] + [SVector(1.0, 2.0) for i=1:100]

function collectleafboxframes(root::BoxTreeNode)
    bboxframe = getboxframe(root.boundingbox)
    bboxframes = typeof(bboxframe)[]

    function collectleafboxframes(node::BoxTreeNode, bboxframe_container)
        if node.children === nothing
            bboxframe = getboxframe(node.boundingbox)
            push!(bboxframe_container, bboxframe)
        else
            for child in node.children
                collectleafboxframes(child, bboxframe_container)
            end
        end
    end
    
    collectleafboxframes(root, bboxframes)

    return bboxframes
end

function plot_boxframes!(plt, boxframes)
    for boxframe in bboxframes
        if size(boxframe,2) == 2
            plot!(plt, boxframe[:,1], boxframe[:,2],    
            line=(2, :black, :solid), 
            label = nothing)
        else
            plot!(plt, boxframe[:,1], boxframe[:,2], boxframe[:,3],  
            line=(2, :black, :solid), 
            label = nothing)
        end
    end
end

tree = create_tree(points2D, BoxTreeOptions(nmin=3))

plt = scatter([points2D[i][1] for i=1:length(points2D)], 
        [points2D[i][2] for i=1:length(points2D)])

bboxframes = collectleafboxframes(tree)

plot_boxframes!(plt, bboxframes)

##
points3D = [@SVector rand(3) for i=1:100] + [SVector(0.0, 0.0, 0.0) for i=1:100]

tree = create_tree(points3D, BoxTreeOptions(nmin=3))

plt = scatter([points3D[i][1] for i=1:length(points3D)], 
        [points3D[i][2] for i=1:length(points3D)], 
        [points3D[i][3] for i=1:length(points3D)],
        markersize = 2)

bboxframes = collectleafboxframes(tree)

plot_boxframes!(plt, bboxframes)
