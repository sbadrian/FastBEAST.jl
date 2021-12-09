points2D = [SVector(1.0, 1.0), #3
            SVector(2.0, 2.0), #1
            SVector(1.2, 1.7), #2
            SVector(1.7, 1.2), #4
            SVector(1.21, 1.7)] #2

root = create_tree(points2D, BoxTreeOptions())

@test length(root.children) == 4
@test root.children[1].data[1] == 2
@test root.children[2].data[1] == 3
@test root.children[2].data[2] == 5
@test root.children[3].data[1] == 1
@test root.children[4].data[1] == 4