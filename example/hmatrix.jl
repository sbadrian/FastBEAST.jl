using FastBEAST
using StaticArrays
using LinearAlgebra
using Plots
using Printf
plotlyjs()
function logkernel(sourcepoint::SVector{2,T}, testpoint::SVector{2,T}) where T
    if isapprox(sourcepoint, testpoint, rtol=eps()*1e1)
        return 0
    else
        return - 2*π*log(norm(sourcepoint - testpoint))
    end
end

N =  100

spoints = [@SVector rand(2) for i = 1:N]

tpoints = 1*[@SVector rand(2) for i = 1:N] + [SVector(0.5, 0.5) for i = 1:N]

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

kmat = assembler(logkernel, spoints, tpoints)

U, S, V = svd(kmat)

println("Condition number: ", S[1] / S[end])

plot(S, yaxis=:log, marker=:x)

##
asmpackage = (assembler, logkernel, spoints, tpoints)
stree = create_tree(spoints, nmin=50)
ttree = create_tree(tpoints, nmin=50)
hmat = HMatrix(asmpackage, stree, stree)


v = rand(N)

@printf("Accuracy test: %.2e", norm(hmat*v - kmat*v)/norm(kmat*v))