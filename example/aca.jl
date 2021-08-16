using FastBEAST
using StaticArrays
using LinearAlgebra
using Printf

function logkernel(sourcepoint::SVector{2,T}, testpoint::SVector{2,T}) where T
    if isapprox(sourcepoint, testpoint, rtol=eps()*1e1)
        return 0
    else
        return - 2*π*log(norm(sourcepoint - testpoint))
    end
end

N =  10000

spoints = [@SVector rand(2) for i = 1:N]

function assembler(kernel, sourcepoints::Vector{SVector{2,T}}, testpoints::Vector{SVector{2,T}}) where T
    kernelmatrix = zeros(promote_type(eltype(testpoints[1]),eltype(sourcepoints[1])), 
                   length(testpoints), length(sourcepoints))

    for i = 1:length(testpoints)
        for j = 1:length(sourcepoints)
            kernelmatrix[i,j] = kernel(testpoints[i], sourcepoints[j])
        end
    end

    return kernelmatrix
end

logkernelassembler(sdata, tdata) = assembler(logkernel, spoints[sdata], spoints[tdata])

stree = create_tree(spoints, nmin=200)
kmat = assembler(logkernel, spoints, spoints)
@time hmat = HMatrix(logkernelassembler, stree, stree, compressor=:aca)

@printf("Compression rate: %.2f %%\n", compressionrate(hmat)*100)

