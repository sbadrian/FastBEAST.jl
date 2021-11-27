using CompScienceMeshes
using BEAST
using LinearAlgebra
using FastBEAST

CM = CompScienceMeshes
function test_beast_laplace_singlelayer(h)
    Γ = CM.meshsphere(1, h)

    X = lagrangecxd0(Γ)
 
    𝒱 = Helmholtz3D.singlelayer(wavenumber=0.0)

    hmat = hassemble(𝒱,X,X, nmin=100)

    return  hmat
end

<<<<<<< HEAD

hmat = test_beast_laplace_singlelayer(0.01) 
=======
hmat = test_beast_laplace_singlelayer(0.1) 
>>>>>>> 233c0a52da7a4dfc0af796d36d9656eef9f1082d
