using Test
using FastBEAST
using CompScienceMeshes
using BEAST
using StaticArrays
using LinearAlgebra
using IterativeSolvers

##

function OneoverRkernel(testpoint::SVector{3,T}, sourcepoint::SVector{3,T}) where T
    if isapprox(testpoint, sourcepoint, rtol=eps()*1e-4)
        return T(0.0)
    else
        return T(1.0) / (norm(testpoint - sourcepoint))
    end
end

function assembler(kernel, matrix, testpoints, sourcepoints)
    for j = 1:length(sourcepoints)
        for i = 1:length(testpoints)
            matrix[i,j] = kernel(testpoints[i], sourcepoints[j])
        end
    end
end

@views OneoverRkernelassembler(matrix, tdata, sdata) = assembler(
    OneoverRkernel,
    matrix,
    spoints[tdata],
    spoints[sdata]
)

## Speed test: only do on a powerful machine
if Threads.nthreads() > 14
    h = 0.02
    Γ = CompScienceMeshes.meshsphere(1.0, h)
    spoints = vertices(Γ)

    stree = create_tree(spoints, BoxTreeOptions(nmin=200))
    stats = @timed (hmatb = HMatrix(
        OneoverRkernelassembler,
        stree,
        stree,
        Int64,
        Float64,
        compressor=:aca,
        svdrecompress=false
    ))

    println("Compression rate (BoxTree): ", compressionrate(hmatb)*100)
    @test compressionrate(hmatb)*100 > 85
    println("Assembly time in s (BoxTree): ", stats.time)
    @test stats.time < 40

    stree = create_tree(spoints, BoxTreeOptions(nmin=200))
    stats = @timed (hmatbs = HMatrix(
        OneoverRkernelassembler,
        stree,
        stree,
        Int64,
        Float64,
        compressor=:aca,
        svdrecompress=true
    ))

    println("Compression rate (BoxTree, SVD): ", compressionrate(hmatbs)*100)
    @test compressionrate(hmatbs)*100 > 88
    println("Assembly time in s (BoxTree, SVD): ", stats.time)
    @test stats.time < 50

    stree = create_tree(spoints, KMeansTreeOptions(nmin=50))
    stats = @timed (hmatk = HMatrix(
        OneoverRkernelassembler,
        stree,
        stree,
        Int64,
        Float64,
        compressor=:aca,
        svdrecompress=false
    ))

    println("Compression rate (KMeans): ", compressionrate(hmatk)*100)
    @test compressionrate(hmatk)*100 > 88
    println("Assembly time in s (KMeans): ", stats.time)
    @test stats.time < 50

    @test estimate_reldifference(hmatb, hmatk) ≈ 0 atol=1e-4

    stree = create_tree(spoints, KMeansTreeOptions(nmin=50))
    stats = @timed (hmatks = HMatrix(
        OneoverRkernelassembler,
        stree,
        stree,
        Int64,
        Float64,
        compressor=:aca,
        svdrecompress=true
    ))

    println("Compression rate (KMeans, SVD): ", compressionrate(hmatks)*100)
    @test compressionrate(hmatks)*100 > 88
    println("Assembly time in s (KMeans, SVD): ", stats.time)
    @test stats.time < 40

    @test estimate_reldifference(hmatbs, hmatks) ≈ 0 atol=1e-4
end

## Speed test: only do on a powerful machine
if Threads.nthreads() > 14

    h = 0.008
    Γ = CompScienceMeshes.meshsphere(1.0, h)
    spoints = vertices(Γ)

    stree = create_tree(spoints, BoxTreeOptions(nmin=200))
    stats = @timed (hmatb = HMatrix(
        OneoverRkernelassembler,
        stree,
        stree,
        Int64,
        Float64,
        compressor=:aca,
        threading=:multi,
        svdrecompress=false
    ))

    println("Compression rate (BoxTree): ", compressionrate(hmatb)*100)
    #@test compressionrate(hmatb)*100 > 85
    println("Assembly time in s (BoxTree): ", stats.time)
    #@test stats.time < 40

    stree = create_tree(spoints, BoxTreeOptions(nmin=200))
    stats = @timed (hmatbs = HMatrix(
        OneoverRkernelassembler,
        stree,
        stree,
        Int64,
        Float64,
        compressor=:aca,
        threading=:multi,
        svdrecompress=true
    ))

    println("Compression rate (BoxTree, SVD): ", compressionrate(hmatbs)*100)
    #@test compressionrate(hmatbs)*100 > 88
    println("Assembly time in s (BoxTree, SVD): ", stats.time)
    #@test stats.time < 50

    stree = create_tree(spoints, KMeansTreeOptions(nmin=50))
    stats = @timed (hmatk = HMatrix(
        OneoverRkernelassembler,
        stree,
        stree,
        Int64,
        Float64,
        compressor=:aca,
        threading=:multi,
        svdrecompress=false
    ))

    println("Compression rate (KMeans): ", compressionrate(hmatk)*100)
    #@test compressionrate(hmatk)*100 > 85
    println("Assembly time in s (KMeans): ", stats.time)
    #@test stats.time < 40

    @test estimate_reldifference(hmatb, hmatk) ≈ 0 atol=1e-4

    stree = create_tree(spoints, KMeansTreeOptions(nmin=50))
    stats = @timed (hmatks = HMatrix(
        OneoverRkernelassembler,
        stree,
        stree,
        Int64,
        Float64,
        compressor=:aca,
        threading=:multi,
        svdrecompress=true
    ))

    println("Compression rate (KMeans, SVD): ", compressionrate(hmatks)*100)
    #@test compressionrate(hmatks)*100 > 88
    println("Assembly time in s (KMeans, SVD): ", stats.time)
    #@test stats.time < 50

    @test estimate_reldifference(hmatbs, hmatks) ≈ 0 atol=1e-4
end