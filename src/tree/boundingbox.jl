using StaticArrays

struct BoundingBox{D,T} 
    halflength::T
    center::SVector{D,T}
end

function BoundingBox(points::Vector{SVector{D,T}}) where {D,T}
    min_dim = Vector(points[1])
    max_dim = Vector(points[1])

    for i = 1:length(points)
        for j = 1:D
            min_dim[j] =  min_dim[j] < points[i][j] ? min_dim[j] : points[i][j]
            max_dim[j] =  max_dim[j] > points[i][j] ? max_dim[j] : points[i][j]
        end
    end

    center = @MVector zeros(D)

    length_dim = zeros(T,D)
    for j = 1:D
        length_dim[j] = max_dim[j] - min_dim[j]
        center[j] = (max_dim[j] + min_dim[j])/2.0
    end

    halflength = maximum(length_dim)/2.0
    
    return BoundingBox(halflength, SVector(center))
end

function getboxframe(bbox::BoundingBox{D,T}) where {D,T}
    if D == 2
        mat = zeros(5, 2)
        sign_x = [1 -1 -1 1 1]
        sign_y = [1 1 -1 -1 1]
        for i = 1:5
            mat[i,1] = bbox.center[1] + sign_x[i]*bbox.halflength 
            mat[i,2] = bbox.center[2] + sign_y[i]*bbox.halflength 
        end
    elseif D == 3
        sign_x = [1 -1 -1  1  1  1  1  1 NaN  1  1 -1 -1 NaN -1 -1 -1 NaN -1  1]
        sign_y = [1  1 -1 -1  1  1 -1 -1 NaN -1 -1 -1 -1 NaN -1  1  1 NaN  1  1]
        sign_z = [1  1  1  1  1 -1 -1  1 NaN -1 -1 -1  1 NaN -1 -1  1 NaN -1 -1]
        mat = zeros(length(sign_x), 3)
        for i = 1:length(sign_x)
            mat[i,1] = bbox.center[1] + sign_x[i]*bbox.halflength 
            mat[i,2] = bbox.center[2] + sign_y[i]*bbox.halflength 
            mat[i,3] = bbox.center[3] + sign_z[i]*bbox.halflength 
        end
    else
        error("Dimension must be 2 or 3")
    end
    return mat
end

"""
    getchildbox(bbox, id)

Returns a bounding box that this is the `id`th quadrants (2D) /
octand (3D) of the bounding box `bbox`
"""
function getchildbox(bbox::BoundingBox{D,T}, id::Integer) where {D,T}

    hl = bbox.halflength / 2.0

    if D == 2
        sign_x = [1 -1 -1  1]
        sign_y = [1  1 -1 -1]

        center = bbox.center + hl .* SVector(sign_x[id], sign_y[id])
    elseif D == 3
        sign_x = [1 -1 -1  1  1 -1 -1  1]
        sign_y = [1  1 -1 -1  1  1 -1 -1]
        sign_z = [1  1  1  1 -1 -1 -1 -1]
        center = bbox.center + hl .* SVector(sign_x[id], sign_y[id],  sign_z[id])
    else
        error("We support only 2 or 3 dimensions")
    end
    return BoundingBox(hl, center)
end

function whichchildbox(bbox::BoundingBox{D,T}, point::SVector{D,T}) where {D,T}

    if point[1] >= bbox.center[1]
        if  point[2] >= bbox.center[2]
            id = 1
        else 
            id = 4
        end
    else
        if point[2] >= bbox.center[2]
            id = 2
        else 
            id = 3
        end
    end

    if D == 3 && point[3] < bbox.center[3]
        id += 4
    end

    return id
end