using Test
using FastBEAST
using LinearAlgebra

matrix = rand(10, 5)
mb = MatrixBlock(matrix, Vector(1:5), Vector(1:10))

##
@test matrix == mb.M

v = rand(5)

@test matrix*v  == mb.M*v

v2 = rand(10)
@test adjoint(matrix)*v2 == adjoint(mb.M)*v2

U = rand(3, 5)
V = rand(5, 10)

lmr = LowRankMatrix(U, V)

@test lmr.U == U
@test lmr.V == V

@test U*(V*v2) == lmr*v2

v3 = rand(3)

@test adjoint(lmr)*v3 == adjoint(V)*(adjoint(U)*v3)

#=
A = rand(100,100)
B = rand(100,100)
@test opnorm(A) ≈ estimate_norm(A) atol=1e-3
@test opnorm(A-B)/opnorm(B) ≈ estimate_reldifference(A,B, tol=1e-6) atol=1e-3
=#