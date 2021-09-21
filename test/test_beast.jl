using CompScienceMeshes
using BEAST
using LinearAlgebra
using FastBEAST
using IterativeSolvers

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

function test_beast_laplace_singlelayer(h; threading=:single, farquaddata=BEAST.quaddata)
    Γ = CM.meshsphere(1, h)

    X = lagrangecxd0(Γ)
 
    𝒱 = Helmholtz3D.singlelayer(wavenumber=0.0)

    hmat = hassemble(𝒱,X,X, nmin=50, threading=threading, farquaddata=farquaddata)

    mat = assemble(𝒱,X,X)
    return mat, hmat
end

mat, hmat_single = test_beast_laplace_singlelayer(0.1) 

@test nnz(hmat_single) == 3917257

@test compressionrate(hmat_single) > 0.3
@test estimate_reldifference(hmat_single, mat) ≈ 0 atol=1e-4

mat, hmat_multi = test_beast_laplace_singlelayer(0.1, threading=:multi) 

@test nnz(hmat_multi) == 3917257

@test compressionrate(hmat_multi) > 0.3
@test estimate_reldifference(hmat_multi, mat) ≈ 0 atol=1e-4
@test compressionrate(hmat) > 0.3

mat, hmat_single = test_beast_laplace_singlelayer(0.1, farquaddata=quaddata) 

@test nnz(hmat_single) == 3917257

@test compressionrate(hmat_single) > 0.3
@test estimate_reldifference(hmat_single, mat) ≈ 0 atol=1e-3

mat, hmat_multi = test_beast_laplace_singlelayer(0.1, threading=:multi, farquaddata=quaddata) 

@test nnz(hmat_multi) == 3917257

@test compressionrate(hmat_multi) > 0.3
@test estimate_reldifference(hmat_multi, mat) ≈ 0 atol=1e-3
@test compressionrate(hmat) > 0.3

#function test_beast_laplace_singlelayer_manufactured(h)

# ##

# mutable struct Monopole{T,P} <: BEAST.Functional
#     location::P
#     wavenumber::T
#     amplitude::T
# end
  
# function Monopole(p,k,a = 1)
#     T = promote_type(eltype(p), typeof(k), typeof(a))
#     P = similar_type(typeof(p), T)
#     Monopole{T,P}(p,k,a)
# end

# monopolehh3d(;
#     location    = error("missing arguement `location`"),
#     wavenumber   = error("missing arguement `wavenumber`"),
#     ) = Monopole(location, wavenumber)


# function (f::Monopole)(r)
#     p = f.location
#     k = f.wavenumber
#     a = f.amplitude
#     return a * 1/norm(r-p)
# end

#     h=0.2
#     Γ = CM.meshsphere(1, h)

#     X = lagrangecxd0(Γ)
    
#     κ = 0.0
#     𝒱 = Helmholtz3D.singlelayer(wavenumber=κ)
#     q = Monopole(SVector(0.35,0.35,0.25), 0.0) #monopolehh3d(location = SVector(0.35,0.35,0.25), wavenumber = 0.0) 
#     f = assemble(strace(q,Γ), X) 
#     hmat = hassemble(𝒱,X,X, nmin=50)
#     mat = assemble(𝒱,X,X)

# ##
#     hx, hch = cg(hmat,f, log=true,reltol=1e-4)

#     x, ch = cg(mat,f, log=true,reltol=1e-4)
# @test norm(hx - x)/norm(x) ≈ 0 atol=1e-3
# ##
#     Φ, Θ = [0.0], range(0,stop=π,length=100)
#     pts = [point(cos(ϕ)*sin(θ), sin(ϕ)*sin(θ), cos(θ)) for ϕ in Φ for θ in Θ]
#     ffd = potential(𝒱, pts, x, X)
    
# #end

# ##
# h=0.02
# Γ = CM.meshsphere(1, h)

# X = lagrangecxd0(Γ)

# κ = 0.0
# 𝒱 = Helmholtz3D.singlelayer(wavenumber=κ)
# hmat = hassemble(𝒱,X,X, nmin=100)
# println(compressionrate(hmat))