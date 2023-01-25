using BEAST
using FastBEAST
using StaticArrays
using CompScienceMeshes
using Test

r = 10.0
λ = 20 * r
k = 2 * π / λ
npoints = 3

refmesh = meshsphere(r, 0.2*r)
S = Helmholtz3D.singlelayer(; gamma=im*k)

# patch basis functions
X0 = lagrangecxd0(refmesh)

fmat = fmmassemble(
    S,
    X0,
    X0,
    nmin=5,
    threading=:multi,
    npoints=npoints,
    fmmncrit=10,
    fmmp=10
);

x = ComplexF64.(rand(Float64, length(X0.fns)))

Ax = fmat * x
Ax_true = assemble(S, X0, X0) * x

println(@test norm(Ax - Ax_true) / norm(Ax_true) ≈ 0 atol=1e-4)

# pyramid basis functions
X0 = lagrangec0d1(refmesh)

fmat = fmmassemble(
    S,
    X0,
    X0,
    nmin=5,
    threading=:multi,
    npoints=npoints,
    fmmncrit=10,
    fmmp=10
);

x = ComplexF64.(rand(Float64, length(X0.fns)))

Ax = fmat * x
Ax_true = assemble(S, X0, X0) * x

println(@test norm(Ax - Ax_true) / norm(Ax_true) ≈ 0 atol=1e-4)

