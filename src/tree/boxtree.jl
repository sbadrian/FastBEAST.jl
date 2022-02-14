struct BoxTreeOptions <: TreeOptions
    nmin
    maxlevel
end

function BoxTreeOptions(; nmin=1, maxlevel=64)
    return BoxTreeOptions(nmin, maxlevel)
end

mutable struct BoxTreeNode{D, I, F, N <: NodeData{I, F}} <: AbstractNode{I, F, N}
    parent::Union{BoxTreeNode{D, I, F, N}, Nothing}
    children::Vector{BoxTreeNode{D, I, F, N}}
    level::I
    boundingbox::BoundingBox{D, F}
    data::N
end

function BoxTreeNode(bbox::BoundingBox{D, F}, data::N) where {D, I, F, N <: NodeData{I, F}}
    return BoxTreeNode(nothing, BoxTreeNode{D, I, F, N}[], I(0), bbox, data)
end

function BoxTreeNode(
    parent::BoxTreeNode{D, I, F, N},
    bbox::BoundingBox{D, F},
    level::I,
    data::N
) where {D, I, F, N <: NodeData{I, F}}

    return BoxTreeNode(parent, BoxTreeNode{D, I, F, N}[], level, bbox, data)
end

function add_child!(
    parent::BoxTreeNode{D, I, F, N},
    bbox::BoundingBox{D, F},
    data::N
) where {D, I, F, N <: NodeData{I, F}}

    childnode = BoxTreeNode(parent, bbox, level(parent) + 1, data)
    push!(parent.children, childnode)    
end

function create_tree(
    points::Vector{SVector{D, F}},
    treeoptions::BoxTreeOptions
) where {D, F}

    root = BoxTreeNode(
        BoundingBox(points),
        ACAData(Vector(1:length(points)), F)
    )

    fill_tree!(root, points, nmin=treeoptions.nmin, maxlevel=treeoptions.maxlevel)
    
    return root
end

function fill_tree!(
    node::BoxTreeNode{D, I, F, N},
    points::Vector{SVector{D, F}};
    nmin=1,
    maxlevel=-log2(eps(eltype(points)))
) where {D, I, F, N  <: NodeData{I, F}}

    if numindices(node) <= nmin || level(node) >= maxlevel - 1
        return
    end

    nchildren = D == 2 ? 4 : 8 # Number of (possible) children
    sorted_points = zeros(I, numindices(node) + 1, nchildren)

    for pindex in indices(node)
        id = whichchildbox(node.boundingbox, points[pindex])
        sorted_points[1, id] +=  1
        sorted_points[1 + sorted_points[1, id], id] = pindex
    end

    for i = 1:nchildren
        if sorted_points[1, i] > 0
            childpointindices = sorted_points[2:(sorted_points[1, i]+1), i]
            add_child!(node, getchildbox(node.boundingbox, i), ACAData(childpointindices, F))
            fill_tree!(lastchild(node), points, nmin=nmin, maxlevel=maxlevel)
        end
    end
end