using BEAST
using CompScienceMeshes
using ExaFMMt
using FastBEAST
using IterativeSolvers
using LinearAlgebra
using MKL
using StaticArrays
using Test

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
Γ = translate(Γ_orig, SVector(-a/2,-a/2,-a/2))

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

𝓣 = Maxwell3D.singlelayer(wavenumber=k)

X = raviartthomas(Γ)

println("Number of RWG functions: ", numfunctions(X))

T = fmmassemble(
    𝓣,
    X,
    X,
    treeoptions= FastBEAST.BoxTreeOptions(nmin=50),
    multithreading=true
)

T_full = assemble(
    𝓣,
    X,
    X
)

e = assemble(𝒆,X)

@test norm(T*e - T_full*e)/norm(T_full*e) ≈ 0 atol=0.01