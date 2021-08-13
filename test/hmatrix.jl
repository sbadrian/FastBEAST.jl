matrix = rand(10,5)
fmatview = FastBEAST.FullMatrixView(matrix, Vector(1:5), Vector(1:10), 10, 5)

@test matrix == fmatview.matrix