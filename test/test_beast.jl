using CompScienceMeshes
using BEAST
using LinearAlgebra
using FastBEAST

CM = CompScienceMeshes

function test_beast_laplace_singlelayer(h)
    Γ = CM.meshsphere(1, h)

    X = lagrangecxd0(Γ)
 
    𝒱 = Helmholtz3D.singlelayer(wavenumber=0.0)

    hmat = hassemble(𝒱,X,X, nmin=50)

    mat = assemble(𝒱,X,X)
    return mat, hmat
end

mat, hmat = test_beast_laplace_singlelayer(0.1) 

@test compressionrate(hmat) > 0.3
@test estimate_reldifference(hmat,mat) ≈ 0 atol=1e-4