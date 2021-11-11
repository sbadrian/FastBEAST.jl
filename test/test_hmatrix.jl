using FastBEAST
using StaticArrays
using LinearAlgebra

# Do a 3D test with Laplace kernel

function OneoverRkernel(testpoint::SVector{3,T}, sourcepoint::SVector{3,T}) where T
    if isapprox(testpoint, sourcepoint, rtol=eps()*1e-4)
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

N =  1000
NT = N

spoints = [@SVector rand(3) for i = 1:N]
tpoints = 0.1*[@SVector rand(3) for i = 1:NT] + [1.0*SVector(3.5, 3.5, 3.5) for i = 1:NT]

@views OneoverRkernelassembler(matrix, tdata, sdata) = assembler(OneoverRkernel, matrix, tpoints[tdata], spoints[sdata])
stree = create_tree(spoints, treeoptions = BoxTreeOptions(nmin=50))
ttree = create_tree(tpoints, treeoptions = BoxTreeOptions(nmin=50))
kmat = assembler(OneoverRkernel, tpoints, spoints)
hmat = HMatrix(OneoverRkernelassembler, ttree, stree, compressor=:naive, T=Float64)

@test estimate_reldifference(hmat,kmat) ≈ 0 atol=1e-4
@test compressionrate(hmat)*100 ≈ 99 atol=1


##
N = 4000
NT = N

spoints = [@SVector rand(3) for i = 1:N]
##
@views OneoverRkernelassembler(matrix, tdata, sdata) = assembler(OneoverRkernel, matrix, spoints[tdata], spoints[sdata])
stree = create_tree(spoints, treeoptions = BoxTreeOptions(nmin=400))
kmat = assembler(OneoverRkernel, spoints, spoints)
hmat = HMatrix(OneoverRkernelassembler, stree, stree, compressor=:aca, T=Float64)

@test estimate_reldifference(hmat,kmat) ≈ 0 atol=1e-4
@test compressionrate(hmat)*100 ≈ 54 atol=1
