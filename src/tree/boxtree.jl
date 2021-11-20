struct BoxTreeOptions <: TreeOptions
    nmin
    maxlevel
end

function BoxTreeOptions(; nmin=1, maxlevel=64)
    return BoxTreeOptions(nmin, maxlevel)
end

mutable struct BoxTreeNode{T} <: AbstractNode
    parent::Union{BoxTreeNode{T},Nothing}
    children::Union{Vector{BoxTreeNode{T}}, Nothing}
    level::Integer
    boundingbox::BoundingBox
    data::T
end

function BoxTreeNode{T}(bbox::BoundingBox, data::T) where T
    return BoxTreeNode{T}(nothing, nothing, 0, bbox, data)
end

function BoxTreeNode{T}(bbox::BoundingBox, level, data::T) where T
    return BoxTreeNode{T}(nothing, nothing, level, bbox, data)
end

function add_child!(parent::BoxTreeNode, bbox::BoundingBox, data::T) where T
    childnode = BoxTreeNode(nothing, nothing, parent.level + 1, bbox, data)
    childnode.parent = parent
    if parent.children === nothing
        parent.children = [childnode]
    else
        push!(parent.children, childnode)
    end
end

function create_tree(
    points::Vector{SVector{D,T}},
    treeoptions::BoxTreeOptions
) where {D,T}
    root = BoxTreeNode(
        nothing, 
        nothing,
        0,
        BoundingBox(points),
        Vector(1:length(points))
    )

    fill_tree!(root, points, nmin=treeoptions.nmin, maxlevel = treeoptions.maxlevel)
    
    return root
end

function fill_tree!(node::BoxTreeNode, points::Vector{SVector{D,T}}; 
                    nmin = 1, maxlevel=-log2(eps(eltype(points)))) where {D,T}

    if length(node.data) <= nmin || node.level >= maxlevel - 1
        return
    end

    nchildren = D == 2 ? 4 : 8 # Number of (possible) children
    sorted_points = zeros(eltype(node.data), length(node.data)+1, nchildren)

    for pindex in node.data
        id = whichchildbox(node.boundingbox, points[pindex])
        sorted_points[1, id] +=  1
        sorted_points[1 + sorted_points[1, id], id] = pindex
    end

    for i = 1:nchildren
        if sorted_points[1, i] > 0
            childpointindices = sorted_points[2:(sorted_points[1, i]+1), i]
            add_child!(node, getchildbox(node.boundingbox, i), childpointindices)
            fill_tree!(node.children[end], points, nmin=nmin, maxlevel=maxlevel)
        end
    end
end