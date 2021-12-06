using FastBEAST
using Pkg
Pkg.add("Documenter")
using Documenter

makedocs(
         sitename = "FastBEAST.jl",
         modules  = [FastBEAST],
         pages=[
                "Home" => "index.md",
                "Manual" => Any[
                    "man/clustering.md"
                    ]
               ])
deploydocs(
    repo="github.com/JoshuaTetzner/FastBEAST",
    devbranch = "main"
)
