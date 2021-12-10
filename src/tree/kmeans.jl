"""
    KMeansTreeOptions <: TreeOptions

Is the datatype that discribes which tree the `create_tree()` function creates.

# Fields
- `iterations`: number of iterations on each level, default is one iterations
- `nchildren`: defines the number of children of each node, default is two
- `nmin`: defines the minimum amount of datapoints which are needed in a 
    cluster so that it gets split up in subclusters, default is 1
- `maxlevel`: defines the maximum amount of levels, default is 100.
"""
struct KMeansTreeOptions <: TreeOptions
    iterations
    nchildren
    nmin
    maxlevel
end

function KMeansTreeOptions(;
    iterations=1,
    nchildren=2,
    nmin=1,
    maxlevel=100
)
    return KMeansTreeOptions(iterations, nchildren, nmin, maxlevel)
end

"""
    KMeansTreeNode{T} <: AbstractNode

Is the datatype of each node in a `K-Means Cluster Tree`.

# Fields
- `parent::Union{KMeansTreeNode{T},Nothing}`: is the superordinate cluster of
    of the represented cluster
- `children::Union{Vector{KMeansTreeNode{T}}, Nothing}`: all directly 
    subordinated clusters of the represented cluster
- `level::Integer`: the level of the represented cluster
- `center`: the center by which the cluster is defined
- `radius`: the euclidian distance between the center and the farthest away
    point
- `data::T`: array containig the indices of the points in this cluster
"""
mutable struct KMeansTreeNode{T} <: AbstractNode
    parent::Union{KMeansTreeNode{T},Nothing}
    children::Union{Vector{KMeansTreeNode{T}}, Nothing}
    level::Integer
    center
    radius
    data::T
end

function KMeansTreeNode{T}(iterations, data::T) where T
    return KMeansTreeNode{T}(nothing, nothing, 0, nothing, 0.0, data)
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

Creates an algebraic tree for an givn set of datapoints. The returned 
datastructure is the foundation for the algorithms in this package. 

# Arguments
- `points::Array{SVector{D, T}, 1}`: is an array of SVectors. Each 
    [SVector](https://juliaarrays.github.io/StaticArrays.jl/stable/pages/api/#SVector-1)
    contains in general two or three float values, which discribe the position 
    in the space.

# Keywords
- `treeoptions::TreeOptions`: this keyword defines by which tree is build. 
    `TreeOptions` is an abstract type which either can be `BoxTreeOptions` or
    [`KMeansTreeOptions`](@ref). Default type is `BoxTreeOptions`.

# Returns
- `AbstractNode`: the root of the created tree. AbstractNode is an abstract type 
    which either can be `BoxTreeNode` or [`KMeansTreeNode`](@ref), depending on the keyword.
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
    if length(node.data) <= nmin|| node.level >= maxlevel - 1 || length(node.data) <= nchildren
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

"""
    iscompressable(sourcenode::KMeansTreeNode, testnode::KMeansTreeNode)

Determins whether two nodes of a tree are comressable. The criteria differs 
between the Boxtree and the K-Means Cluster Tree.
For the K Means Cluster Tree two nodes can be compressed, if the distance 
between the centers of two clusters is greater than the sum of their radius 
multiplied by a factor of 1.5.
For the Boxtree two nodes can be compressed, if the distance between the centers 
of the two boxes is greater than the sum of the distances of each box's center to 
one of its corners multiplied by a factor of 1.1.

# Arguments
- `sourcenode::AbstractNode`: the node which is observed
- `testnode::AbstractNode`: the node which is tested for compression
"""
function iscompressable(sourcenode::KMeansTreeNode, testnode::KMeansTreeNode)
    if sourcenode.level > 0 && testnode.level > 0
        dist = norm(sourcenode.center - testnode.center)
        factor = 1.5
        if factor * (sourcenode.radius + testnode.radius) < dist
            return true
        else
            return false
        end
    else
        return false
    end
end