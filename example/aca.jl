using FastBEAST
using StaticArrays
using LinearAlgebra

function OneoverRkernel(testpoint::SVector{3,T}, sourcepoint::SVector{3,T}) where T
    if isapprox(testpoint, sourcepoint, rtol=eps()*1e1)
        return 0.0
    else
        return 1.0 / (norm(testpoint - sourcepoint))
    end
end

function assembler(kernel, testpoints, sourcepoints)
    kernelmatrix = zeros(
        promote_type(eltype(testpoints[1]),eltype(sourcepoints[1])), 
        length(testpoints),
        length(sourcepoints)
    )

    for i in eachindex(testpoints)
        for j in eachindex(sourcepoints)
            kernelmatrix[i,j] = kernel(testpoints[i], sourcepoints[j])
        end
    end

    return kernelmatrix
end

function assembler(kernel, matrix, testpoints, sourcepoints)
    for i in eachindex(testpoints)
        for j in eachindex(sourcepoints)
            matrix[i,j] = kernel(testpoints[i], sourcepoints[j])
        end
    end
end

N=100
spoints = [@SVector rand(3) for i = 1:N]
tpoints = 0.1*[@SVector rand(3) for i = 1:N] + [SVector(2.0, 0.0, 0.0) for i = 1:N]
@views OneoverRkernelassembler(matrix, tdata, sdata) = assembler(
    OneoverRkernel, matrix, tpoints[tdata], spoints[sdata]
)
lm = LazyMatrix(OneoverRkernelassembler, Vector(1:N), Vector(1:N), Float64)
U, V = aca(lm; tol=1e-4)