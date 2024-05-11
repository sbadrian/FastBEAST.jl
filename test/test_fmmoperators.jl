using BEAST
using CompScienceMeshes
using ExaFMMt
using FastBEAST
using IterativeSolvers
using LinearAlgebra
using MKL
using StaticArrays
using Test

r = 10.0
λ = 20 * r
k = 2 * π / λ

sphere = meshsphere(r, 0.2 * r)
X0 = lagrangecxd0(sphere)
X1 = lagrangec0d1(sphere)
Y1 = duallagrangec0d1(sphere)
##
Os = [
    (Helmholtz3D.singlelayer(;), Y1, X1)
    (Helmholtz3D.doublelayer(;), Y1, X1)
    (Helmholtz3D.doublelayer_transposed(;), Y1, X1)
    (Helmholtz3D.hypersingular(;), Y1, X1)
    (Helmholtz3D.singlelayer(; alpha=30.0), Y1, X1)
    (Helmholtz3D.doublelayer(; alpha=30.0), Y1, X1)
    (Helmholtz3D.doublelayer_transposed(; alpha=30.0), Y1, X1)
    (Helmholtz3D.hypersingular(; alpha=30.0), Y1, X1)
    (Helmholtz3D.hypersingular(; alpha=0.0, beta=-100.0), Y1, X1)
    (Helmholtz3D.singlelayer(; gamma=3.0), Y1, X1) # Does not work with dual testing functions
    (Helmholtz3D.doublelayer(; gamma=3.0), Y1, X1) # Does not work with dual testing functions
    (Helmholtz3D.doublelayer_transposed(; gamma=3.0), Y1, X1) # Does not work with dual testing functions
    (Helmholtz3D.hypersingular(; gamma=3.0), Y1, X1) # Does not work with dual testing functions
    (Helmholtz3D.singlelayer(; wavenumber=k), Y1, X1)
    (Helmholtz3D.doublelayer(; wavenumber=k), Y1, X1)
    (Helmholtz3D.doublelayer_transposed(; wavenumber=k), Y1, X1)
    (Helmholtz3D.hypersingular(; wavenumber=k), Y1, X1)
    (Helmholtz3D.singlelayer(; alpha=3.0im, gamma=3.0), Y1, X1)
    (Helmholtz3D.doublelayer(; alpha=3.0im, gamma=3.0), Y1, X1)
    (Helmholtz3D.doublelayer_transposed(; alpha=3.0im, gamma=3.0), Y1, X1)
    (Helmholtz3D.hypersingular(; alpha=3.0im, beta=1/2.0im, gamma=3.0), Y1, X1)
]

@testset "FMM mvp test: $O" for (O, Y1, X1) in Os
    @show O
    Oft = fmmassemble(O, Y1, X1, treeoptions=BoxTreeOptions(nmin=20), fmmoptions=ExaFMMOptions(p=8), computetransposeadjoint=true) # fast
    Ofl = assemble(O, Y1, X1) # full

    x = rand(numfunctions(X1))

    yt = Oft*x
    yl = Ofl*x
    @test eltype(yt) == promote_type(eltype(x), eltype(Oft)) 
    @test norm(yt - yl)/norm(yl) ≈ 0 atol=1e-2

    x = rand(numfunctions(Y1))

    yt = transpose(Oft)*x
    yl = transpose(Ofl)*x
    @test eltype(yt) == promote_type(eltype(x), eltype(Oft)) 
    @test norm(yt - yl)/norm(yl) ≈ 0 atol=1e-2

    yt = adjoint(Oft)*x
    yl = adjoint(Ofl)*x
    @test eltype(yt) == promote_type(eltype(x), eltype(Oft)) 
    @test norm(yt - yl)/norm(yl) ≈ 0 atol=1e-2
end