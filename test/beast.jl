using CompScienceMeshes
using BEAST
using LinearAlgebra
using FastBEAST

CM = CompScienceMeshes
##
function test_beast_laplace_singlelayer(h)
    Γ = CM.meshsphere(1, h) # CM.read_gmsh_mesh(fn)

    X = lagrangecxd0(Γ)
    @show numfunctions(X)

    # Compute system matrix here

    𝒱 = Helmholtz3D.singlelayer(wavenumber=0.0)

    @views blkasm = BEAST.blockassembler(𝒱,X,X)
    
    @views function singlelayerassembler(Z, tdata, sdata)
        @views store(v,m,n) = (Z[m,n] += v)
        blkasm(tdata,sdata,store)
    end

    ##

    stree = create_tree(X.pos, nmin=50)
    @time hmat = HMatrix(singlelayerassembler, stree, stree, 
                         compressor=:svd, T=Float64, tol=1e-4)


    mat = assemble(𝒱,X,X)
    return mat, hmat
end

mat, hmat = test_beast_laplace_singlelayer(0.1) 

@test estimate_reldifference(hmat,mat) ≈ 0 atol=1e-4