using CompScienceMeshes
using BEAST
using LinearAlgebra
using FastBEAST

CM = CompScienceMeshes

function farquaddata(op::BEAST.Helmholtz3DOp, test_refspace::BEAST.LagrangeRefSpace,
    trial_refspace::BEAST.LagrangeRefSpace, test_elements, trial_elements)

    test_eval(x)  = test_refspace(x,  Val{:withcurl})
    trial_eval(x) = trial_refspace(x, Val{:withcurl})

    # The combinations of rules (6,7) and (5,7 are) BAAAADDDD
    # they result in many near singularity evaluations with any
    # resemblence of accuracy going down the drain! Simply don't!
    # (same for (5,7) btw...).
    # test_qp = quadpoints(test_eval,  test_elements,  (6,))
    # bssi_qp = quadpoints(trial_eval, trial_elements, (7,))

    test_qp = quadpoints(test_eval,  test_elements,  (1,))
    bsis_qp = quadpoints(trial_eval, trial_elements, (1,))

    return test_qp, bsis_qp
end

function test_beast_laplace_singlelayer(h)
    Γ = CM.meshsphere(1, h)

    X = lagrangecxd0(Γ)
 
    𝒱 = Helmholtz3D.singlelayer(wavenumber=0.0)

    hmat = hassemble(𝒱,X,X, nmin=100, farquaddata=farquaddata)

    return  hmat
end

hmat = test_beast_laplace_singlelayer(0.01) 
