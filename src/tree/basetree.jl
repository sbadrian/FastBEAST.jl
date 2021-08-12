abstract type AbstractNode end

mutable struct Node{T} <: AbstractNode
    parent::Union{Node{T},Nothing}
    children::Union{Vector{Node{T}}, Nothing}
    data::T

    Node{T}() where T = new{T}(nothing, nothing, nothing)
    Node{T}(data::T) where T = new{T}(nothing, nothing, data)
end


mutable struct BoxTreeNode{T} <: AbstractNode
    parent::Union{BoxTreeNode{T},Nothing}
    children::Union{Vector{BoxTreeNode{T}}, Nothing}
    data::T
end

function BoxTreeNode{T}(data::T) where T
    return BoxTreeNode{T}(nothing, nothing, data)
end

function add_child!(parent::BoxTreeNode, data::T) where T
    childnode = BoxTreeNode(nothing, nothing, data)
    childnode.parent = parent
    if parent.children == nothing
        parent.children = [childnode]
    else
        push!(parent.children, childnode)
    end
end


function create_tree(points::Vector{SVector{D,T}}) where {D,T}
    bbox = BoundingBox(points)

    root = BoxTreeNode(nothing, nothing, Vector(1:length(points)))
    nchildren = D == 2 ? 4 : 8 # Number of (possible) children
    sorted_points = zeros(eltype(root.data), length(points)+1, nchildren)

    for i = 1:length(points)
        id = whichchildbox(bbox, points[i])
        sorted_points[1, id] +=  1
        sorted_points[1 + sorted_points[1, id], id] = i
    end

    for i = 1:nchildren
        if sorted_points[1, i] > 0
            childpointindices = sorted_points[2:(sorted_points[1, i]+1), i]
            add_child!(root, childpointindices)
        end
    end

    return root
end