using CompScienceMeshes
using BEAST
using LinearAlgebra
using FastBEAST

CM = CompScienceMeshes
function test_beast_laplace_singlelayer(h)
    Γ = CM.meshsphere(1, h)

    X = lagrangecxd0(Γ)
 
    𝒱 = Helmholtz3D.singlelayer(wavenumber=0.0)

    hmat = hassemble(𝒱,X,X, treeoptions = BoxTreeOptions(nmin=100))

    return  hmat
end

hmat = test_beast_laplace_singlelayer(0.1) 
