using Test
using BEAST
using CompScienceMeshes
using StaticArrays
using BenchmarkTools
using LinearAlgebra
using FastBEAST

@testset "ACA vs BEAST" begin
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

    ##
    U, V = aca_compression(
        assembler,
        Vector(1:numfunctions(X1)),
        Vector(1:numfunctions(X2)),
        Float64,
        tol=1e-14,
        maxrank=100,
        svdrecompress=false
    )

    rank_k = size(U, 2)

    bm_aca =  @benchmark U, V = aca_compression(
        $assembler,
        $(Vector(1:numfunctions(X1))),
        $(Vector(1:numfunctions(X2))),
        Float64,
        tol=1e-14,
        maxrank=100,
        svdrecompress=false
    )

    ##
    Z = zeros(Float64, numfunctions(X1), numfunctions(X2))

    @time assembler(Z, Vector(1:numfunctions(X1)), Vector(1:numfunctions(X2)))

    @test norm(Z-U*V)/norm(Z) < 1e-13

    VV = zeros(Float64, rank_k, numfunctions(X2))

    UU = zeros(Float64, numfunctions(X1), rank_k)

    ##

    function oneshoot(dim, UU, VV)
        assembler(VV, 1:dim, Vector(1:numfunctions(X2)))
        assembler(UU, Vector(1:numfunctions(X1)), 1:dim)
    end

    bm_oneshoot = @benchmark oneshoot($rank_k, $UU, $VV)

    function iteratively(dim, UU, VV)
        for i=1:dim
            assembler(VV, i, Vector(1:numfunctions(X2)))
            assembler(UU, Vector(1:numfunctions(X1)), i)
        end
    end

    bm_iteratively = @benchmark iteratively($rank_k, $UU, $VV)

    ##

    #Roughly 10% overhead in terms of memory use
    rel_aca_memory = norm(bm_aca.memory - bm_iteratively.memory)/norm(bm_iteratively.memory)
    @test rel_aca_memory ≈ 0.09 atol=0.01
    rel_aca_allocs = norm(bm_aca.allocs - bm_iteratively.allocs)/norm(bm_iteratively.allocs)
    @test rel_aca_allocs ≈ 0.24 atol=0.01
    rel_aca_times = norm(median(bm_aca.times) - median(bm_iteratively.times))/norm(median(bm_iteratively.times))
    @test rel_aca_times ≈ 0.15 atol=0.01

    rel_oneshoot_memory = norm(bm_oneshoot.memory - bm_iteratively.memory)/norm(bm_iteratively.memory)
    @test rel_oneshoot_memory ≈ 0.26 atol=0.01
    rel_oneshoot_allocs = norm(bm_oneshoot.allocs - bm_iteratively.allocs)/norm(bm_iteratively.allocs)
    @test rel_oneshoot_allocs ≈ 0.003 atol=0.001
    rel_oneshoot_times = norm(median(bm_oneshoot.times) - median(bm_iteratively.times))/norm(median(bm_iteratively.times))
    @test rel_oneshoot_times ≈ 0.12 atol=0.01

    optimality_aca_memory = norm(rel_aca_memory - rel_oneshoot_memory)/norm(rel_oneshoot_memory)
    @test optimality_aca_memory ≈ 0.6 atol=0.1
    optimality_aca_allocs = norm(rel_aca_allocs - rel_oneshoot_allocs)/norm(rel_oneshoot_allocs)
    @test optimality_aca_allocs ≈ 65 atol=1
    optimality_aca_times = norm(rel_aca_times - rel_oneshoot_times)/norm(rel_oneshoot_times)
    @test optimality_aca_times ≈ 0.19 atol=0.01

end