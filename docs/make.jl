using FastBEAST
using StaticArrays
using Pkg
Pkg.add("Documenter")
using Documenter

makedocs(
         sitename = "FastBEAST.jl",
         authors="Simon B. Adrian <simon.adrian@fake.de> and contributors",
         modules=[FastBEAST],
         pages=[
                "Introduction" => "index.md",
                "Manual" => Any[
                    "General Usage" => "./manual/manual.md",
                    "Application Examples" => "./manual/application.md"
                ],
                "Fast Methods" => Any[
                    "H-Matrix" => "./fastmethods/hmatrix.md",
                    #"H2-Matrix" => "./fastmethods/h2matrix.md",
                    "FMM" => "./fastmethods/fmm.md"],
                "Low Rank Approximations" => Any[
                    "Adaptive Cross Approximation" => "./lowrankapprox/aca.md",
                ],
                "Clustering" => "clustering.md",
                "Contributing" => "contributing.md",
                "API Reference" => "apiref.md"
               ])
deploydocs(
    repo="github.com/sbadrian/FastBEAST.jl.git",
    devbranch = "doc/documentation",
    versions = nothing,
)
