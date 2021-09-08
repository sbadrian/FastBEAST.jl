using StaticArrays

points2D = [SVector(-1.0, -1.0), SVector(1.0, 0.5), SVector(0.5, 1.0) , SVector(0.2, 0.3) ]

bbox = BoundingBox(points2D)

@test bbox.halflength == 1.0
@test bbox.center == SVector(0.0,0.0)

points2D = [SVector(1.0, 2.0), SVector(2.0, 3.0), SVector(1.5, 2.0) , SVector(1.8, 2.8) ]

bbox = BoundingBox(points2D)

@test bbox.halflength == 0.5
@test bbox.center == SVector(1.5, 2.5)

@test getchildbox(bbox, 3) == BoundingBox(0.25, SVector(1.25, 2.25))

@test whichchildbox(bbox, SVector(1.0, 2.0)) == 3
@test whichchildbox(bbox, SVector(2.0, 3.0)) == 1
@test whichchildbox(bbox, SVector(1.5, 2.0)) == 4
@test whichchildbox(bbox, SVector(1.8, 2.8)) == 1