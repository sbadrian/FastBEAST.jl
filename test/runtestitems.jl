using TestItems
using Test

using FastBEAST
using StaticArrays

##

@testitem "BoundingBox" tags=[:fast] begin
    include("test_boundingbox.jl")
end

@testitem "BoxTree" tags=[:fast] begin
    include("test_boxtree.jl")
end

@testitem "KMeans" tags=[:fast] begin
    include("test_kmeans.jl")
end

@testitem "Skeletons" tags=[:fast] begin
    include("test_skeletons.jl")
end

@testitem "ACA" tags=[:fast] begin
    include("test_aca.jl")
end

@testitem "ACA vs BEAST Blockassembler" tags=[:fast] begin
    #include("test_aca_beast.jl")
end

@testitem "HMatrix" begin
    include("test_hmatrix.jl")
end

@testitem "ACA + BEAST" tags=[:fast] begin
    include("test_beast.jl")
end

@testitem "ACA + BEAST for EFIE" tags=[:fast] begin
    include("test_beast_efie.jl")
end

@testitem "FMM Bases Test" tags=[:fast] begin
    include("test_fmmbases.jl")
end

@testitem "FMM Operators" tags=[:fast] begin
    include("test_fmmoperators.jl")
end

@testitem "ExaFMM EFIE" tags=[:fast] begin
    include("test_fmmefie.jl")
end

@testitem "ExaFMM MFIE" tags=[:fast] begin
    include("test_fmmmfie.jl")
end
