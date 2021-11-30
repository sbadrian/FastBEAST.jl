struct KMeansTreeOptions <: TreeOptions
    iterations
    nchildren
    nmin
    maxlevel
end

function KMeansTreeOptions(;
    iterations = 1,
    nchildren=2,
    nmin=1,
    maxlevel=100
)
    return KMeansTreeOptions(iterations, nchildren, nmin, maxlevel)
end

mutable struct KMeansTreeNode{T} <: AbstractNode
    parent::Union{KMeansTreeNode{T},Nothing}
    children::Union{Vector{KMeansTreeNode{T}}, Nothing}
    level::Integer
    center
    radius
    data::T
end

function KMeansTreeNode{T}(iterations, data::T) where T
    return KMeansTreeNode{T}(nothing, nothing, 0, nothing, 0.0,data)
end

function KMeansTreeNode{T}(level, center::Vector{Float64}, radius,data::T) where T
    return KMeansTreeNode{T}(nothing, nothing, level, center, radius, data)
end

function add_child!(parent::KMeansTreeNode, center, radius, data::T) where T
    childnode = KMeansTreeNode(nothing, nothing, parent.level + 1, center, radius, data)
    childnode.parent = parent
    if parent.children === nothing
        parent.children = [childnode]
    else
        push!(parent.children, childnode)
    end
end

"""
    create_tree(points::Array{SVector{D, T}, 1}; treeoptions)

Creates a tree for a given set of points. The treetyp is defined by the 
abstract type treeoptions.
Treeoptions can be the KMeansTreeOptions resulting in a tree with the KMeans 
sorting criteria or BoxTreeOptions, resulting in a quad- (2D) or octtree(3D)
sorted in equal sized Boxes. 

`KMeansTreeOptions` takes the variables:
  * `iterations`: number of iterations on each level, which increases the uniformity of the Tree (default=1).
  * `nchildren`: defines the number of children of each node (default=2)
  * `nmin`: defines the minimum amount of points in each box (default=1).
  * `malevel`: defines the maximum amount of levels (default=100).

`BoxTreeOptions` takes the variables 
  * `nmin`: defines the minimum amount of points in each box (default=1).
  * `maxlevel`: defines the maximum amount of levels (default=100).
"""
function create_tree(
    points::Vector{SVector{D,T}},
    treeoptions::KMeansTreeOptions
) where {D,T}
    root = KMeansTreeNode( nothing, nothing,  0, nothing, 0.0, Vector(1:length(points)))
    fill_tree!(
        root,
        points,
        nmin=treeoptions.nmin,
        maxlevel=treeoptions.maxlevel,
        iterations=treeoptions.iterations,
        nchildren=treeoptions.nchildren
    )

    return root
end

function whichcenter(center, point, nchildren)
    centerindex = argmin(norm.([center[i]-point for i = 1:nchildren]))

    return centerindex
end

function fill_tree!(
    node::KMeansTreeNode, points::Vector{SVector{D,T}}; 
    nmin = 1,
    maxlevel=log2(eps(eltype(points))),
    iterations, 
    nchildren
) where {D,T}
    if length(node.data) <= nmin*nchildren || node.level >= maxlevel - 1 || length(node.data) <= nchildren
        return
    end

    center = points[node.data[1:nchildren]]
    sorted_points = zeros(eltype(node.data), length(node.data)+1, nchildren)

    for iter = 1:iterations
        sorted_points .= 0
        for pindex in node.data
            id = whichcenter(center, points[pindex], nchildren)
            sorted_points[1, id] +=  1
            sorted_points[1 + sorted_points[1, id], id] = pindex
        end
        for k=1:nchildren
            center[k] = sum(
                points[sorted_points[2:(sorted_points[1, k]+1), k]]
            ) ./ sorted_points[1,k]          
        end
    end
    

    for i = 1:nchildren
        if sorted_points[1, i] > 0
            childpointindices = sorted_points[2:(sorted_points[1, i]+1), i]
            radius = maximum(norm.([points[childpointindices[j]].-center[i] for j = 1:length(childpointindices)]))
            add_child!(node, center[i], radius, childpointindices)
            fill_tree!(
                node.children[end], 
                points, 
                nmin=nmin, 
                maxlevel=maxlevel, 
                iterations=iterations, 
                nchildren=nchildren
            )
        end
    end
end

function iscompressable(sourcenode::KMeansTreeNode, testnode::KMeansTreeNode)
    if sourcenode.level > 0 && testnode.level > 0
        dist = norm(sourcenode.center - testnode.center)
        factor = 1.1
        if factor * (sourcenode.radius + testnode.radius) < dist
            return true
        else
            return false
        end
    else
        return false
    end
end