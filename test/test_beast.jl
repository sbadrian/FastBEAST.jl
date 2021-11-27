using CompScienceMeshes
using BEAST
using LinearAlgebra
using FastBEAST
using IterativeSolvers

CM = CompScienceMeshes

function test_beast_laplace_singlelayer(h; threading=:single, 
    farquaddata=nothing, svdrecompress=false)

    Γ = CM.meshsphere(1, h)

    X = lagrangecxd0(Γ)
 
    𝒱 = Helmholtz3D.singlelayer(wavenumber=0.0)

    hmat = hassemble(𝒱,X,X, BoxTreeOptions(nmin=50), threading=threading, svdrecompress=svdrecompress)

    mat = assemble(𝒱,X,X)
    return mat, hmat
end

mat, hmat_single = test_beast_laplace_singlelayer(0.1) 

@test nnz(hmat_single) == 3916760

@test compressionrate(hmat_single) > 0.3
@test estimate_reldifference(hmat_single, mat) ≈ 0 atol=1e-4

mat, hmat_multi = test_beast_laplace_singlelayer(0.1, threading=:multi) 

@test nnz(hmat_multi) == 3916760

@test compressionrate(hmat_multi) > 0.3
@test estimate_reldifference(hmat_multi, mat) ≈ 0 atol=1e-4
@test compressionrate(hmat) > 0.3

mat, hmat_single = test_beast_laplace_singlelayer(0.1, farquaddata=quaddata) 

@test nnz(hmat_single) == 3916760

@test compressionrate(hmat_single) > 0.3
@test estimate_reldifference(hmat_single, mat) ≈ 0 atol=1e-3

mat, hmat_multi = test_beast_laplace_singlelayer(0.1, threading=:multi, farquaddata=quaddata) 

@test nnz(hmat_multi) == 3916760

@test compressionrate(hmat_multi) > 0.3
@test estimate_reldifference(hmat_multi, mat) ≈ 0 atol=1e-3
@test compressionrate(hmat) > 0.3

mat, hmat_svdmulti = test_beast_laplace_singlelayer(0.1, threading=:multi, farquaddata=quaddata, svdrecompress=true) 

@test nnz(hmat_svdmulti) == 3356811

@test compressionrate(hmat_svdmulti) > 0.3
@test estimate_reldifference(hmat_svdmulti, mat) ≈ 0 atol=1e-3
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