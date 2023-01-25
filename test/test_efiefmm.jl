using Test
using MKL
using CompScienceMeshes
using BEAST
using ExaFMMt
using LinearAlgebra
using FastBEAST

r = 10.0
λ = 20 * r
k = 2 * π / λ

sphere = meshsphere(r, 0.5 * r)
X = raviartthomas(sphere)

##
S = Maxwell3D.doublelayer(; wavenumber=k)

x = ComplexF64.(rand(Float64, length(X.fns)))

Ax = fmmassemble(
    S,
    X,
    X,
    threading=:multi,
    fmmoptions=HelmholtzFMMOptions(ComplexF64(k))
) * x

Axtrue = assemble(S, X, X) * x

norm(Ax.-Axtrue)/norm(Axtrue)

## Singlelayer
S = Maxwell3D.singlelayer(; wavenumber=k)

x = ComplexF64.(rand(Float64, length(X.fns)))

Ax = fmmassemble(
    S,
    X,
    X,
    threading=:multi,
    fmmoptions=HelmholtzFMMOptions(ComplexF64(k))
) * x

Axtrue = assemble(S, X, X) * x

norm(Ax.-Axtrue)/norm(Axtrue)


##

using Test
using FastBEAST
using CompScienceMeshes
using BEAST
using StaticArrays
using LinearAlgebra
using IterativeSolvers

c = 3e8
μ = 4*π*1e-7
ε = 1/(μ*c^2)
f = 1e8
λ = c/f
k = 2*π/λ
ω = k*c
η = sqrt(μ/ε)

a = 1.0
Γ_orig = CompScienceMeshes.meshcuboid(a,a,a,0.2)
Γ = translate(Γ_orig,SVector(-a/2,-a/2,-a/2))

Φ, Θ = [0.0], range(0,stop=π,length=100)
pts = [point(cos(ϕ)*sin(θ), sin(ϕ)*sin(θ), cos(θ)) for ϕ in Φ for θ in Θ]

# This is an electric dipole
# The pre-factor (1/ε) is used to resemble 
# (9.18) in Jackson's Classical Electrodynamics
E = (1/ε) * dipolemw3d(location=SVector(0.4,0.2,0), 
                    orientation=1e-9.*SVector(0.5,0.5,0), 
                    wavenumber=k)

n = BEAST.NormalVector()

𝒆 = (n × E) × n
H = (-1/(im*μ*ω))*curl(E)
𝒉 = (n × H) × n

𝓣 = Maxwell3D.singlelayer(wavenumber=k)
𝓝 = BEAST.NCross()
𝓚 = Maxwell3D.doublelayer(wavenumber=k)

X = raviartthomas(Γ)

println("Number of RWG functions: ", numfunctions(X))

T = fmmassemble(
    𝓣,
    X,
    X,
    threading=:multi,
    fmmoptions=HelmholtzFMMOptions(ComplexF64(k))
)

e = assemble(𝒆,X)
##
println("Enter iterative solver")
@time j_EFIE, ch = IterativeSolvers.gmres(T, e, verbose=true, log=true, reltol=1e-4, maxiter=500)
println("Finished iterative solver part. Number of iterations: ", ch.iters)

nf_E_EFIE = potential(MWSingleLayerField3D(wavenumber=k), pts, j_EFIE, X)
nf_H_EFIE = potential(BEAST.MWDoubleLayerField3D(wavenumber=k), pts, j_EFIE, X) ./ η
ff_E_EFIE = potential(MWFarField3D(wavenumber=k), pts, j_EFIE, X)

@test norm(nf_E_EFIE - E.(pts))/norm(E.(pts)) ≈ 0 atol=0.01
@test norm(nf_H_EFIE - H.(pts))/norm(H.(pts)) ≈ 0 atol=0.01
@test norm(ff_E_EFIE - E.(pts, isfarfield=true))/norm(E.(pts, isfarfield=true)) ≈ 0 atol=0.01