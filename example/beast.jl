using CompScienceMeshes
using BEAST
using Printf
using LinearAlgebra
using FastBEAST

CM = CompScienceMeshes
##
function test_beast_laplace_singlelayer(h)
    Γ = CM.meshsphere(1, h) # CM.read_gmsh_mesh(fn)


    @show CM.numcells(Γ)
    @show CM.numvertices(Γ)

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

    stree = create_tree(X.pos, nmin=400)
    @time hmat = HMatrix(singlelayerassembler, stree, stree, compressor=:aca, T=Float64)

    @printf("Compression rate: %.2f %%\n", compressionrate(hmat)*100)
    return hmat
end

test_beast_laplace_singlelayer(0.06) 

##
function test_beast_efie(h)
    Γ = CM.meshsphere(1, h) # CM.read_gmsh_mesh(fn)


    @show CM.numcells(Γ)
    @show CM.numvertices(Γ)

    X = raviartthomas(Γ)
    @show numfunctions(X)

    # Compute system matrix here
    κ = 5.0; γ = κ*im;
    𝒯 = Maxwell3D.singlelayer(gamma=γ)
    
    @views blkasm = BEAST.blockassembler(𝒯,X,X)
    
    @views function efieassembler(Z, tdata, sdata)
        @views store(v,m,n) = (Z[m,n] += v)
        blkasm(tdata,sdata,store)
    end

    ##
    stree = create_tree(X.pos, nmin=400)
    @time hmat = HMatrix(efieassembler, stree, stree, compressor=:aca, T=ComplexF64)

    @printf("Compression rate: %.2f %%\n", compressionrate(hmat)*100)
    return hmat
end

test_beast_efie(0.036)

# assembler = blockassembler(op, tfs, bfs)
# function μ2(τ,σ)
#     Z = zeros(T,length(τ),length(σ))
#     assembler(τ,σ,(v,m,n)->(Z[m,n] += v))
#     return Z
# end