using BEAST
using CompScienceMeshes
using ExaFMMt
using FastBEAST
using IterativeSolvers
using LinearAlgebra
using MKL
using Test

r = 10.0
λ = 20 * r
k = 2 * π / λ

sphere = meshsphere(r, 0.5 * r)

X0 = lagrangecxd0(sphere)
X1 = lagrangec0d1(sphere)
Xd0 = duallagrangecxd0(sphere)
Xd1 = duallagrangec0d1(sphere)

S = Helmholtz3D.singlelayer(; wavenumber=k)

xX0 = rand(ComplexF64, length(X0.fns))
xX1 = rand(ComplexF64, length(X1.fns))
xXd0 = rand(ComplexF64, length(Xd0.fns))
xXd1 = rand(ComplexF64, length(Xd1.fns))

yX0 = fmmassemble(
    S,
    X0,
    X0,
    threading=:multi,
    nmin=10,
    fmmoptions=HelmholtzFMMOptions(ComplexF64(k))
) * xX0

yX1 = fmmassemble(
    S,
    X1,
    X1,
    threading=:multi,
    fmmoptions=HelmholtzFMMOptions(ComplexF64(k))
) * xX1

yXd0 = fmmassemble(
    S,
    Xd0,
    Xd0,
    threading=:multi,
    fmmoptions=HelmholtzFMMOptions(ComplexF64(k))
) * xXd0

yXd1 = fmmassemble(
    S,
    Xd1,
    Xd1,
    threading=:multi,
    fmmoptions=HelmholtzFMMOptions(ComplexF64(k))
) * xXd1

yX0_true = assemble(S, X0, X0) * xX0
yX1_true = assemble(S, X1, X1) * xX1
yXd0_true = assemble(S, Xd0, Xd0) * xXd0
yXd1_true = assemble(S, Xd1, Xd1) * xXd1

@test norm(yX0 - yX0_true) /norm(yX0_true) < 1e-5
@test norm(yX1 - yX1_true) /norm(yX1_true) < 1e-5
@test norm(yXd0 - yXd0_true) /norm(yXd0_true) < 1e-5
@test norm(yXd1 - yXd1_true) /norm(yXd1_true) < 1e-5
