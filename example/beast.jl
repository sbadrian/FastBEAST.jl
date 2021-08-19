using CompScienceMeshes
using BEAST
using Printf
using LinearAlgebra
using FastBEAST

CM = CompScienceMeshes
##
function test_beast_aca(h)
    Γ = CM.meshcuboid(1,1,1,h) # CM.read_gmsh_mesh(fn)


    @show CM.numcells(Γ)
    @show CM.numvertices(Γ)

    X = lagrangecxd0(Γ)

    println("Number of functions: ", numfunctions(X))
    # Compute system matrix here

    𝒱 = Helmholtz3D.singlelayer(wavenumber=0.0)


    @views function singlelayerassembler(Z, tdata, sdata)
        #Xt = subset(X,tdata)
        #Xs = subset(X,sdata)

        #return assemble(𝒱,Xt,Xs)

        store(v,m,n) = (Z[m,n] += v)

        blkasm = BEAST.blockassembler(𝒱,X,X)
        blkasm(tdata,sdata,store)
    end

    #

    stree = create_tree(X.pos, nmin=400)
    @time hmat = HMatrix(singlelayerassembler, stree, stree, compressor=:aca)

    #
    #if numfunctions(X) <= 2000
    #    𝗩 = assemble(𝒱,X,X)
    #end

    @printf("Compression rate: %.2f %%\n", compressionrate(hmat)*100)
    return hmat
end