using Test
using FastBEAST
using StaticArrays
using LinearAlgebra

# Do a 3D test with Laplace kernel

function OneoverRkernel(testpoint::SVector{3,T}, sourcepoint::SVector{3,T}) where T
    if isapprox(testpoint, sourcepoint, rtol=eps()*1e-4)
        return T(0.0)
    else
        return T(1.0) / (norm(testpoint - sourcepoint))
    end
end

function assembler(kernel, testpoints, sourcepoints)
    kernelmatrix = zeros(promote_type(eltype(testpoints[1]),eltype(sourcepoints[1])), 
                length(testpoints), length(sourcepoints))

    for j = 1:length(sourcepoints)
        for i = 1:length(testpoints)
            kernelmatrix[i,j] = kernel(testpoints[i], sourcepoints[j])
        end
    end
    return kernelmatrix
end


function assembler(kernel, matrix, testpoints, sourcepoints)
    for j = 1:length(sourcepoints)
        for i = 1:length(testpoints)
            matrix[i,j] = kernel(testpoints[i], sourcepoints[j])
        end
    end
end

##

N =  1000
NT = N

spoints = [@SVector rand(3) for i = 1:N]
tpoints = 0.1*[@SVector rand(3) for i = 1:NT] + [1.0*SVector(3.5, 3.5, 3.5) for i = 1:NT]

@views OneoverRkernelassembler(matrix, tdata, sdata) = assembler(OneoverRkernel, matrix, tpoints[tdata], spoints[sdata])
stree = create_tree(spoints, BoxTreeOptions(nmin=50))
ttree = create_tree(tpoints, BoxTreeOptions(nmin=50))
kmat = assembler(OneoverRkernel, tpoints, spoints)
hmat = HMatrix(OneoverRkernelassembler, ttree, stree, Int64, Float64, compressor=:naive)

@test estimate_reldifference(hmat,kmat) ≈ 0 atol=1e-4
@test compressionrate(hmat)*100 ≈ 99 atol=1

##
N = 4000
NT = N

spoints = [@SVector rand(3) for i = 1:N]

v = rand(N)
##

@views OneoverRkernelassembler(matrix, tdata, sdata) = assembler(
    OneoverRkernel,
    matrix,
    spoints[tdata],
    spoints[sdata]
)
stree = create_tree(spoints, KMeansTreeOptions(nmin=20))
@time kmat = assembler(OneoverRkernel, spoints, spoints)
@time hmat = HMatrix(
    OneoverRkernelassembler,
    stree,
    stree,
    Int64,
    Float64,
    compressor=:aca,
    svdrecompress=false
)

@test estimate_reldifference(hmat, kmat) ≈ 0 atol=1e-4
@test 18 < compressionrate(hmat)*100 < 26

##
@time hmat = HMatrix(
    OneoverRkernelassembler,
    stree,
    stree,
    Int64,
    Float64,
    compressor=:aca,
    svdrecompress=true
)

@test estimate_reldifference(hmat, kmat) ≈ 0 atol=1e-4
@test 29 < compressionrate(hmat)*100 < 37

##
@views OneoverRkernelassembler(matrix, tdata, sdata) = assembler(
    OneoverRkernel,
    matrix,
    spoints[tdata],
    spoints[sdata]
)
stree = create_tree(spoints, BoxTreeOptions(nmin=400))
@time kmat = assembler(OneoverRkernel, spoints, spoints)
@time hmat = HMatrix(
    OneoverRkernelassembler,
    stree,
    stree,
    Int64,
    Float64,
    compressor=:aca,
    svdrecompress=false
)

@test estimate_reldifference(hmat,kmat) ≈ 0 atol=1e-4
@test 45 < compressionrate(hmat)*100 < 50

@time hmat = HMatrix(
    OneoverRkernelassembler,
    stree,
    stree,
    Int64,
    Float64,
    compressor=:aca,
    svdrecompress=true
)

@test estimate_reldifference(hmat,kmat) ≈ 0 atol=1e-4
@test 47 < compressionrate(hmat)*100 < 57

##

@time hmat = HMatrix(
    OneoverRkernelassembler,
    stree,
    stree,
    Int64,
    Float64,
    compressor=:aca,
    svdrecompress=false
)

@test estimate_reldifference(hmat,kmat) ≈ 0 atol=1e-4
@test 45 < compressionrate(hmat)*100 < 50

@time hmatm = HMatrix(
    OneoverRkernelassembler,
    stree,
    stree,
    Int64,
    Float64,
    compressor=:aca,
    threading=:multi,
    svdrecompress=false
)

@test hmat*v ≈ hmatm*v
@test transpose(hmat)*v ≈ transpose(hmatm)*v
@test adjoint(hmat)*v ≈ adjoint(hmatm)*v

println("Large test, N=40000")

## Speed test: only do on a powerful machine
if Threads.nthreads() > 13
    N = 40000

    spoints = [@SVector rand(3) for i = 1:N]
    
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
        svdrecompress=true
    ))

    println("Compression rate (BoxTree, SVD): ", compressionrate(hmatbs)*100)
    #@test compressionrate(hmatbs)*100 > 88
    println("Assembly time in s (BoxTree, SVD): ", stats.time)
    #@test stats.time < 50

    stree = create_tree(spoints, KMeansTreeOptions(nmin=30))
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
    #@test 56 < compressionrate(hmatk)*100 < 57
    println("Assembly time in s (KMeans): ", stats.time)
    #@test stats.time < 60

    @test estimate_reldifference(hmatb, hmatk) ≈ 0 atol=1e-4

    stree = create_tree(spoints, KMeansTreeOptions(nmin=30))
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
    #@test 59 < compressionrate(hmatks)*100 < 61
    println("Assembly time in s (KMeans, SVD): ", stats.time)
    #@test 45 < stats.time < 80

    @test estimate_reldifference(hmatbs, hmatks) ≈ 0 atol=1e-4
end

println("Large test, N=100000")

## Speed test: only do on a powerful machine
if Threads.nthreads() > 13
    N = 100000

    spoints = [@SVector rand(3) for i = 1:N]

    stree = create_tree(spoints, BoxTreeOptions(nmin=200))
    stats = @timed (hmat = HMatrix(
        OneoverRkernelassembler,
        stree,
        stree,
        Int64,
        Float64,
        compressor=:aca,
        svdrecompress=false
    ))

    println("Compression rate (BoxTree): ", compressionrate(hmat)*100)
    @test compressionrate(hmat)*100 > 90
    println("Assembly time in s (BoxTree): ", stats.time)
    @test stats.time < 90

    stree = create_tree(spoints, BoxTreeOptions(nmin=200))
    stats = @timed (hmat = HMatrix(
        OneoverRkernelassembler,
        stree,
        stree,
        Int64,
        Float64,
        compressor=:aca,
        svdrecompress=true
    ))

    println("Compression rate (BoxTree, SVD): ", compressionrate(hmat)*100)
    @test compressionrate(hmat)*100 > 90
    println("Assembly time in s (BoxTree, SVD): ", stats.time)
    @test stats.time < 120
end
