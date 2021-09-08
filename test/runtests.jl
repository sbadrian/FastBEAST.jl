using FastBEAST
using Test

@testset "FastBEAST.jl" begin
    include("boundingbox.jl")
    include("boxtree.jl")
    include("skeletons.jl")
    include("aca.jl")
    include("hmatrix_laplace_kernel_3d.jl")
    #include("beast.jl")
end
