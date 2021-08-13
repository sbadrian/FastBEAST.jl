using FastBEAST
using Test

@testset "FastBEAST.jl" begin
    include("boundingbox.jl")
    include("boxtree.jl")
    include("hmatrix.jl")
end
