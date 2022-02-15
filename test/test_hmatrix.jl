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
@time hmat = HMatrix(OneoverRkernelassembler, stree, stree, Int64, Float64, compressor=:aca)

@test estimate_reldifference(hmat, kmat) ≈ 0 atol=1e-4
@test 29 < compressionrate(hmat)*100 < 34

##
@views OneoverRkernelassembler(matrix, tdata, sdata) = assembler(
    OneoverRkernel,
    matrix,
    spoints[tdata],
    spoints[sdata]
)
stree = create_tree(spoints, BoxTreeOptions(nmin=400))
@time kmat = assembler(OneoverRkernel, spoints, spoints)
@time hmat = HMatrix(OneoverRkernelassembler, stree, stree, Int64, Float64, compressor=:aca)

@test estimate_reldifference(hmat,kmat) ≈ 0 atol=1e-4
@test 52 < compressionrate(hmat)*100 < 57


@time hmatm = HMatrix(
    OneoverRkernelassembler,
    stree,
    stree,
    Int64,
    Float64,
    compressor=:aca,
    threading=:multi
)

@test hmat*v ≈ hmatm*v
@test transpose(hmat)*v ≈ transpose(hmatm)*v
@test adjoint(hmat)*v ≈ adjoint(hmatm)*v

## Speed test: only do on a powerful machine
if Threads.nthreads() > 13
    N = 100000
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
    stree = create_tree(spoints, BoxTreeOptions(nmin=200))
    stats = @timed HMatrix(
        OneoverRkernelassembler,
        stree,
        stree,
        Int64,
        Float64,
        compressor=:aca
    )

    println("Compression rate (BoxTree): ", compressionrate(hmat)*100)
    println("BoxTree assembly time in s: ", stats.time)
    @test stats.time < 120
end

## Speed test: only do on a powerful machine
if Threads.nthreads() > 13
    N = 40000
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
    stree = create_tree(spoints, KMeansTreeOptions(nmin=50))
    stats = @timed (hmat = HMatrix(
        OneoverRkernelassembler,
        stree,
        stree,
        Int64,
        Float64,
        compressor=:aca
    ))

    println("Compression rate (KMeans): ", compressionrate(hmat)*100)
    println("KMeans assembly time in s: ", stats.time)
    @test stats.time < 120
end