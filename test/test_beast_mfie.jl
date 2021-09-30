using CompScienceMeshes
using BEAST
using StaticArrays
using LinearAlgebra
using IterativeSolvers
using FastBEAST
using Test

function farquaddata(op::BEAST.MaxwellOperator3D,
    test_local_space::BEAST.RefSpace, trial_local_space::BEAST.RefSpace,
    test_charts, trial_charts)

    a, b = 0.0, 1.0
    # CommonVertex, CommonEdge, CommonFace rules

    tqd = quadpoints(test_local_space, test_charts, (1,6))
    bqd = quadpoints(trial_local_space, trial_charts, (1,7))
    leg = (BEAST._legendre(3,a,b), BEAST._legendre(4,a,b), BEAST._legendre(5,a,b),)

    # High accuracy rules (use them e.g. in LF MFIE scenarios)
    # tqd = quadpoints(test_local_space, test_charts, (8,8))
    # bqd = quadpoints(trial_local_space, trial_charts, (8,9))
    # leg = (_legendre(8,a,b), _legendre(10,a,b), _legendre(5,a,b),)


    return (tpoints=tqd, bpoints=bqd, gausslegendre=leg)
end


c = 3e8
μ = 4*π*1e-7
ε = 1/(μ*c^2)
f = 1e8
λ = c/f
k = 2*π/λ
ω = k*c
η = sqrt(μ/ε)

a = 1
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
Y = buffachristiansen(Γ)

println("Number of RWG functions: ", numfunctions(X))

K_bc = hassemble(𝓚,Y,X,
                nmin=10,
                threading=:multi,
                verbose=true,
                farquaddata=farquaddata,
                svdrecompress=false,
                dblsupport=true)


G_nxbc_rt = Matrix(assemble(𝓝,Y,X))
h_bc = η*Vector(assemble(𝒉,Y))
K_bc_full = assemble(𝓚,Y,X)
M_bc = -0.5*G_nxbc_rt + K_bc

##
# Note with farquaddata, we will have larger errors than 1e-4.
#=
for (i, mv) in enumerate(K_bc.matrixviews)
    if norm(K_bc_full[mv.leftindices, mv.rightindices] - mv.leftmatrix*mv.rightmatrix) > 1e-3
        println("i: ", i)
    end
end =#
##
println("Enter iterative solver")
@time j_BCMFIE, ch = IterativeSolvers.gmres(M_bc, h_bc, log=true, reltol=1e-4, maxiter=500)
#println("Finished iterative solver part. Number of iterations: ", ch.iters)
##


nf_E_BCMFIE = potential(MWSingleLayerField3D(wavenumber=k), pts, j_BCMFIE, X)
nf_H_BCMFIE = potential(BEAST.MWDoubleLayerField3D(wavenumber=k), pts, j_BCMFIE, X) ./ η
ff_E_BCMFIE = potential(MWFarField3D(wavenumber=k), pts, j_BCMFIE, X)

@test norm(nf_E_BCMFIE - E.(pts))/norm(E.(pts)) ≈ 0 atol=0.01
@test norm(nf_H_BCMFIE - H.(pts))/norm(H.(pts)) ≈ 0 atol=0.01
@test norm(ff_E_BCMFIE - E.(pts, isfarfield=true))/norm(E.(pts, isfarfield=true)) ≈ 0 atol=0.01



#T = hassemble(𝓣,X,X, 
#                nmin=100, 
#                threading=:multi, 
#                farquaddata=farquaddata, 
#                verbose=true, 
#                svdrecompress=true)
##
'