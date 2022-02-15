using Test
using BEAST
using CompScienceMeshes
using StaticArrays
using BenchmarkTools
using LinearAlgebra
using FastBEAST

CM = CompScienceMeshes

h = 0.1
Γ1 = CM.meshsphere(1, h)
Γ2 = translate(Γ1, SVector(10.0,0.0,0.0))
X1 = lagrangecxd0(Γ1)
X2 = lagrangecxd0(Γ2)

𝒱 = Helmholtz3D.singlelayer(wavenumber=0.0)

blkasm = BEAST.blockassembler(𝒱, X1, X2, quadstrat=BEAST.DoubleNumQStrat(1,1))
    
function assembler(Z, tdata, sdata)
    store(v,m,n) = (Z[m,n] += v)
    blkasm(tdata,sdata,store)
end

function oneshoot(dim, UU, VV, X1, X2)
    assembler(VV, 1:dim, Vector(1:numfunctions(X2)))
    assembler(UU, Vector(1:numfunctions(X1)), 1:dim)
end

lm = LazyMatrix(assembler, Vector(1:numfunctions(X1)), Vector(1:numfunctions(X2)), Float64)

function iteratively(dim, UU, VV, X1, X2)
    for i=1:dim
        assembler(VV, i, Vector(1:numfunctions(X2)))
        assembler(UU, Vector(1:numfunctions(X1)), i)
    end
end

am = allocate_aca_memory(Float64, numfunctions(X1), numfunctions(X2), maxrank=100)

##
U, V = aca_compression(
    lm,
    am,
    tol=1e-14,
    maxrank=100,
    svdrecompress=false
)

rank_k = size(U, 2)

bm_aca =  @benchmark U, V = aca_compression(
    $lm,
    $am,
    tol=1e-14,
    maxrank=100,
    svdrecompress=false
)

Z = zeros(Float64, numfunctions(X1), numfunctions(X2))

@time assembler(Z, Vector(1:numfunctions(X1)), Vector(1:numfunctions(X2)))

@test norm(Z-U*V)/norm(Z) < 1e-13

VV = zeros(Float64, rank_k, numfunctions(X2))

UU = zeros(Float64, numfunctions(X1), rank_k)

bm_oneshoot = @benchmark oneshoot($rank_k, $UU, $VV, $X1, $X2)
bm_iteratively = @benchmark iteratively($rank_k, $UU, $VV, $X1, $X2)


#Roughly 10% overhead in terms of memory use
@show rel_aca_memory = bm_aca.memory/bm_oneshoot.memory
@test rel_aca_memory ≈ 5.02 atol=0.01
@show rel_aca_allocs = bm_aca.allocs/bm_oneshoot.allocs
@test rel_aca_allocs ≈ 1.01 atol=0.01
@show rel_aca_times = median(bm_aca.times)/median(bm_oneshoot.times)
@test 2.0 < rel_aca_times < 2.6

@show rel_iter_memory = bm_iteratively.memory/bm_oneshoot.memory
@test rel_iter_memory ≈ 5.01 atol=0.1
@show rel_iter_allocs = bm_iteratively.allocs/bm_oneshoot.allocs
@test rel_iter_allocs ≈ 1.014 atol=0.001
@show rel_iter_times = median(bm_iteratively.times)/median(bm_oneshoot.times)
@test 1.4 < rel_iter_times < 1.9

@show optimality_aca_memory = rel_aca_memory/rel_iter_memory
@test optimality_aca_memory ≈ 1.000 atol=0.001
@show optimality_aca_allocs = rel_aca_allocs/rel_iter_allocs
@test optimality_aca_allocs ≈ 1.00 atol=0.01
@show optimality_aca_times = rel_aca_times/rel_iter_times
@test 1.1 < optimality_aca_times < 1.5

##
U, V = aca_compression(
    lm,
    am,
    tol=1e-2,
    maxrank=100,
    svdrecompress=false
)

rank_k = size(U, 2)

bm_aca =  @benchmark U, V = aca_compression(
    $lm,
    $am,
    tol=1e-2,
    maxrank=100,
    svdrecompress=false
)

Z = zeros(Float64, numfunctions(X1), numfunctions(X2))

@time assembler(Z, Vector(1:numfunctions(X1)), Vector(1:numfunctions(X2)))
@show 
@test norm(Z-U*V)/norm(Z) < 1e-2

VV = zeros(Float64, rank_k, numfunctions(X2))

UU = zeros(Float64, numfunctions(X1), rank_k)

bm_oneshoot = @benchmark oneshoot($rank_k, $UU, $VV, $X1, $X2)
bm_iteratively = @benchmark iteratively($rank_k, $UU, $VV, $X1, $X2)


#Roughly 10% overhead in terms of memory use
@show rel_aca_memory = bm_aca.memory/bm_oneshoot.memory
@test rel_aca_memory ≈ 2.19 atol=0.01
@show rel_aca_allocs = bm_aca.allocs/bm_oneshoot.allocs
@test rel_aca_allocs ≈ 1.01 atol=0.01
@show rel_aca_times = median(bm_aca.times)/median(bm_oneshoot.times)
@test 1.4 < rel_aca_times < 1.8

@show rel_iter_memory = bm_iteratively.memory/bm_oneshoot.memory
@test rel_iter_memory ≈ 2.19 atol=0.01
@show rel_iter_allocs = bm_iteratively.allocs/bm_oneshoot.allocs
@test rel_iter_allocs ≈ 1.009 atol=0.001
@show rel_iter_times = median(bm_iteratively.times)/median(bm_oneshoot.times)
@test 1.25 < rel_iter_times < 1.65

@show optimality_aca_memory = rel_aca_memory/rel_iter_memory
@test optimality_aca_memory ≈ 1.000 atol=0.001
@show optimality_aca_allocs = rel_aca_allocs/rel_iter_allocs
@test optimality_aca_allocs ≈ 1.00 atol=0.01
@show optimality_aca_times = rel_aca_times/rel_iter_times
@test 1.03 < optimality_aca_times < 1.4
