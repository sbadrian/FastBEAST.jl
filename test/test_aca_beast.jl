using Test
using FastBEAST
using BEAST
using CompScienceMeshes
using StaticArrays
using BenchmarkTools

CM = CompScienceMeshes

h = 0.5
Γ1 = CM.meshsphere(1, h)
Γ2 = translate(Γ1, SVector(10.0,0.0,0.0))
X1 = lagrangecxd0(Γ1)
X2 = lagrangecxd0(Γ2)

𝒱 = Helmholtz3D.singlelayer(wavenumber=0.0)

blkasm = BEAST.blockassembler(𝒱, X1, X2)
    
function assembler(Z, tdata, sdata)
    store(v,m,n) = (Z[m,n] += v)
    blkasm(tdata,sdata,store)
end

Z = zeros(Float64, numfunctions(X1), numfunctions(X2))

lm = LazyMatrix(assembler, Vector(1:numfunctions(X1)), Vector(1:numfunctions(X1)), Float64)

##

@benchmark U, V = aca_compression(lm, maxrank=100, tol=1e-14, svdrecompress=false)