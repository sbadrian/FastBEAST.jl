using FastBEAST
using StaticArrays
using LinearAlgebra

using Printf

function OneoverRkernel(testpoint::SVector{3,T}, sourcepoint::SVector{3,T}) where T
    if isapprox(testpoint, sourcepoint, rtol=eps()*1e1)
        return 0.0
    else
        return 1.0 / (norm(testpoint - sourcepoint))
    end
end

function assembler(kernel, testpoints, sourcepoints)
    kernelmatrix = zeros(promote_type(eltype(testpoints[1]),eltype(sourcepoints[1])), 
                length(testpoints), length(sourcepoints))

    for i = 1:length(testpoints)
        for j = 1:length(sourcepoints)
            kernelmatrix[i,j] = kernel(testpoints[i], sourcepoints[j])
        end
    end
    return kernelmatrix
end


function assembler(kernel, matrix, testpoints, sourcepoints)
    for i = 1:length(testpoints)
        for j = 1:length(sourcepoints)
            matrix[i,j] = kernel(testpoints[i], sourcepoints[j])
        end
    end
end

##

N =  100
NT = N

spoints = [@SVector rand(3) for i = 1:N]
tpoints = 0.1*[@SVector rand(3) for i = 1:NT] + [1.0*SVector(3.5, 3.5, 3.5) for i = 1:NT]

@views OneoverRkernelassembler(matrix, tdata, sdata) = assembler(OneoverRkernel, matrix, tpoints[tdata], spoints[sdata])
stree = create_tree(spoints, nmin=50)
ttree = create_tree(tpoints, nmin=50)
kmat = assembler(OneoverRkernel, tpoints, spoints)
hmat = HMatrix(OneoverRkernelassembler, ttree, stree, compressor=:naive)

@printf("Accuracy test: %.2e\n", estimate_reldifference(hmat,kmat))
@printf("Compression rate: %.2f %%\n", compressionrate(hmat)*100)


##
N = 1000
NT = N

spoints = [@SVector rand(3) for i = 1:N]
##
@views OneoverRkernelassembler(matrix, tdata, sdata) = assembler(OneoverRkernel, matrix, spoints[tdata], spoints[sdata])
stree = create_tree(spoints, nmin=100)
kmat = assembler(OneoverRkernel, spoints, spoints)
hmat = HMatrix(OneoverRkernelassembler, stree, stree, compressor=:aca, isdebug=false)

@printf("Accuracy test: %.2e\n", estimate_reldifference(hmat,kmat))
@printf("Compression rate: %.2f %%\n", compressionrate(hmat)*100)

##
totalspoints = 0
for child in stree.children
    totalspoints += length(child.data)
end

@assert totalspoints == N


##



function hmatrix3d_benchmark(N)
    spoints = [@SVector rand(3) for i = 1:N]
    
    @views OneoverRkernelassembler(matrix, tdata, sdata) = assembler(OneoverRkernel, matrix, spoints[tdata], spoints[sdata])
    stree = create_tree(spoints, nmin=400)
    
    @time hmat = HMatrix(OneoverRkernelassembler, stree, stree, compressor=:aca, tol=1e-4, isdebug=false)
    
    @printf("Compression rate: %.2f %%\n", compressionrate(hmat)*100)
    return hmat;
end


