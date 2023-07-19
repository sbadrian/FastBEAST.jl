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
H = (-1/(im*μ*ω))*curl(E)
𝒉 = (n × H) × n

𝓣 = Maxwell3D.singlelayer(wavenumber=k)
𝓝 = BEAST.NCross()
𝓚 = Maxwell3D.doublelayer(wavenumber=k)

X = raviartthomas(Γ)
Y = buffachristiansen(Γ)

println("Number of RWG functions: ", numfunctions(X))

K_bc = fmmassemble(
    𝓚,
    HelmholtzFMMOptions(ComplexF64(-k)),
    Y,
    X,
    treeoptions=FastBEAST.BoxTreeOptions(nmin=30),
    multithreading=true    
)

G_nxbc_rt = Matrix(assemble(𝓝,Y,X))
h_bc = η*Vector(assemble(𝒉,Y))
K_bc_full = assemble(𝓚,Y,X)
M_bc = -0.5*G_nxbc_rt + K_bc

println("Enter iterative solver")
@time j_BCMFIE, ch = IterativeSolvers.gmres(
    M_bc,
    h_bc,
    verbose=true,
    log=true,
    reltol=1e-4,
    maxiter=500
)

nf_E_BCMFIE = potential(MWSingleLayerField3D(wavenumber=k), pts, j_BCMFIE, X)
nf_H_BCMFIE = potential(BEAST.MWDoubleLayerField3D(wavenumber=k), pts, j_BCMFIE, X) ./ η
ff_E_BCMFIE = potential(MWFarField3D(wavenumber=k), pts, j_BCMFIE, X)

@test norm(nf_E_BCMFIE - E.(pts))/norm(E.(pts)) ≈ 0 atol=0.01
@test norm(nf_H_BCMFIE - H.(pts))/norm(H.(pts)) ≈ 0 atol=0.01
@test norm(ff_E_BCMFIE - E.(pts, isfarfield=true))/norm(E.(pts, isfarfield=true)) ≈ 0 atol=0.01
