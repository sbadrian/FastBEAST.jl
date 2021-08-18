matrix = rand(10,5)
fmv = FastBEAST.FullMatrixView(matrix, Vector(1:5), Vector(1:10), 10, 5)

@test matrix == fmv.matrix

v = rand(5)

@test matrix*v  == fmv*v

adjointfmv = adjoint(fmv)
adjoint_adjointfmv = adjoint(adjointfmv)

@test adjoint_adjointfmv == fmv

v2 = rand(10)
@test adjoint(matrix)*v2 == adjoint(fmv)*v2

rightmatrix = rand(5,10)
leftmatrix = rand(3,5)

lmv = FastBEAST.LowRankMatrixView(rightmatrix, leftmatrix, Vector(1:10), Vector(1:3), 3, 10)

@test lmv.rightmatrix == rightmatrix
@test lmv.leftmatrix == leftmatrix

@test lmv*v2  ==  leftmatrix*(rightmatrix*v2)

v3 = rand(3)

@test adjoint(lmv)*v3  ==  adjoint(rightmatrix)*(adjoint(leftmatrix)*v3)

A = rand(100,100)
B = rand(100,100)
using LinearAlgebra
@test opnorm(A) ≈ estimate_norm(A) atol=1e-3
@test opnorm(A-B)/opnorm(B) ≈ estimate_reldifference(A,B) atol=1e-3
