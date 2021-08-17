using FastBEAST
using StaticArrays
using LinearAlgebra

using Printf

function OneoverRkernel(sourcepoint::SVector{3,T}, testpoint::SVector{3,T}) where T
    if isapprox(sourcepoint, testpoint, rtol=eps()*1e1)
        return 0.0
    else
        return 1.0 / (norm(sourcepoint - testpoint))
    end
end


function assembler(kernel, sourcepoints, testpoints)
    kernelmatrix = zeros(promote_type(eltype(testpoints[1]),eltype(sourcepoints[1])), 
                length(testpoints), length(sourcepoints))

    for i = 1:length(testpoints)
        for j = 1:length(sourcepoints)
            kernelmatrix[i,j] = kernel(testpoints[i], sourcepoints[j])
        end
    end

    return kernelmatrix
end


##

N =  100
NT = N

spoints = [@SVector rand(3) for i = 1:N]
tpoints = 0.1*[@SVector rand(3) for i = 1:NT] + [1.0*SVector(3.5, 3.5, 3.5) for i = 1:NT]

OneoverRkernelassembler(sdata, tdata) = assembler(OneoverRkernel, spoints[sdata], tpoints[tdata])
stree = create_tree(spoints, nmin=50)
ttree = create_tree(tpoints, nmin=50)
kmat = assembler(OneoverRkernel, spoints, tpoints)
hmat = HMatrix(OneoverRkernelassembler, stree, ttree, compressor=:naive)

@printf("Accuracy test: %.2e\n", estimate_reldifference(hmat,kmat))
@printf("Compression rate: %.2f %%\n", compressionrate(hmat)*100)


##
N = 4000
NT = N

spoints = [@SVector rand(3) for i = 1:N]
##
OneoverRkernelassembler(sdata, tdata) = assembler(OneoverRkernel, spoints[sdata], spoints[tdata])
stree = create_tree(spoints, nmin=100)
kmat = assembler(OneoverRkernel, spoints, spoints)
hmat = HMatrix(OneoverRkernelassembler, stree, stree, compressor=:aca)

@printf("Accuracy test: %.2e\n", estimate_reldifference(hmat,kmat))
@printf("Compression rate: %.2f %%\n", compressionrate(hmat)*100)


totalspoints = 0
for child in stree.children
    totalspoints += length(child.data)
end

@assert totalspoints == N