using FastBEAST
using Test

@testset "FastBEAST.jl" begin
    include("test_boundingbox.jl")
    include("test_boxtree.jl")
    include("test_skeletons.jl")
    include("test_aca.jl")
    include("test_hmatrix.jl")
    include("test_beast.jl")
end