using FastBEAST
using Pkg
Pkg.add("Documenter")
using Documenter

makedocs(
         sitename = "FastBEAST.jl",
         modules  = [FastBEAST],
         pages=[
                "Home" => "index.md",
                "Getting Started" => "gstarted.md",
                "Manual" => Any[
                    "man/clustering.md",
                    "man/hmatrix.md",
                    "man/aca.md"
                    ],
                "Types and Functions" => "functions.md"
               ])
deploydocs(
    repo="github.com/sbadrian/FastBEAST",
    devbranch = "main"
)
