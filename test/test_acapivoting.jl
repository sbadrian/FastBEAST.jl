using Base.Threads
using BEAST
using CompScienceMeshes
using FastBEAST
using IterativeSolvers
using LinearAlgebra
using StaticArrays
using Test

# random distribution
function greensfunction(
    src::Vector{SVector{3, F}},
    trg::Vector{SVector{3, F}}
) where {F <: Real}

    G = zeros(F, size(trg)[1], size(src)[1])

    @threads for row = 1:size(trg)[1]
        @threads for col = 1:size(src)[1]
        
            if src[row, :] != trg[col, :]
                r = norm(src[row, :] - trg[col, :])
                G[row, col] = 1/(4*pi*r)
            end
        end
    end

    return G
end

nsrc = 1000
ntrg = 1000
s = [@SVector rand(3) for i = 1:nsrc]
t = [@SVector rand(3) for i = 1:ntrg] + 1.5 .* [@SVector ones(3) for i = 1:ntrg]

src = s
trg = t
A = greensfunction(src, trg)

function fct(C, x, y)
    for i in eachindex(x)
        for j in eachindex(y)
            C[i,j] = A[x[i],y[j]]
        end
    end
end

rowindices = Array(1:nsrc)
colindices = Array(1:ntrg)

lm = LazyMatrix(fct, rowindices, colindices, Float64);
fd = FastBEAST.FillDistance(t)
tfd = FastBEAST.TrueFillDistance(t)

# classic aca filldistance
@time U, V = aca(lm, rowpivstrat=fd, maxrank=100, tol=1e-6);
@show size(U)
@test norm(U*V - A) / norm(A) ≈ 0 atol=1e-5

# adaptive aca
@time U, V = aca(lm, fd, maxrank=100, tol=1e-6);
@show size(U)
@test norm(U*V - A) / norm(A) ≈ 0 atol=1e-5

# classic aca max pivoting
@time U, V = aca(lm, maxrank=100, tol=1e-6);
@show size(U)
@test norm(U*V - A) / norm(A) ≈ 0 atol=1e-5

## inhomo Plates
c = 3e8
f = 1e8
λ = c/f
k = 2*π/λ

fn = joinpath(@__DIR__, "geometries/ihplate.msh")
src = CompScienceMeshes.read_gmsh_mesh(fn)
trg = CompScienceMeshes.translate(src, SVector(-2.5, -1.0, 0.0))
trg = CompScienceMeshes.rotate(trg, SVector(0.0,0.0,pi))

MS = Maxwell3D.singlelayer(wavenumber=k, alpha=0.0im)
XS = raviartthomas(src)
XT = raviartthomas(trg)

@views farblkasm = BEAST.blockassembler(
    MS,
    XT,
    XS,
    quadstrat=BEAST.defaultquadstrat(MS, XT, XS)
)

@views function farassembler(Z, tdata, sdata)
    @views store(v,m,n) = (Z[m,n] += v)
    farblkasm(tdata,sdata,store)
end

test_tree = create_tree(XT.pos, FastBEAST.BoxTreeOptions(nmin=1000))
trial_tree = create_tree(XS.pos, FastBEAST.BoxTreeOptions(nmin=1000))

A = assemble(MS, XT, XS)
lm = LazyMatrix(farassembler, FastBEAST.indices(test_tree), FastBEAST.indices(trial_tree), scalartype(MS))

fd = FastBEAST.FillDistance(XS.pos)

# classic aca filldistance
@time U, V = aca(lm, rowpivstrat=fd, tol=1e-4, svdrecompress=false);
@show size(U)
@test norm(U*V - A) / norm(A) ≈ 0 atol=1e-3

# adaptive aca
@time U, V = aca(lm, fd, maxrank=100, tol=1e-4);
@show size(U)
@test norm(U*V - A) / norm(A) ≈ 0 atol=1e-3

# classic aca max pivoting
@time U, V = aca(lm, maxrank=100, tol=1e-4, svdrecompress=false);
@show size(U)
@test norm(U*V - A) / norm(A) ≈ 0 atol=1e-3

##
c = 3e8
f = 1e8
λ = c/f
k = 2*π/λ

## alternately strucutred
src = CompScienceMeshes.meshrectangle(1.0, 1.0, 0.4, 3)
src2 = CompScienceMeshes.rotate(src, SVector(0, pi/2, 0))
src2 = CompScienceMeshes.rotate(src2, SVector(0, 0, pi))
src2 = CompScienceMeshes.translate(src2, SVector(0, 1, 0))

trg = CompScienceMeshes.translate(src, SVector(2.0, 0, 0))
Γsrc = CompScienceMeshes.weld(src, src2)
Γtrg = trg

## block strucutred
src = CompScienceMeshes.meshrectangle(1.0, 1.0, 0.4, 3)
src2 = CompScienceMeshes.rotate(src, SVector(0, pi/2, 0))
src2 = CompScienceMeshes.rotate(src2, SVector(0, 0, pi))
src2 = CompScienceMeshes.translate(src2, SVector(-1, 1,-1))

Γsrc = CompScienceMeshes.weld(src, src2)
Γtrg = CompScienceMeshes.translate(Γsrc, SVector(0, 2, 0))

##
using Plotly
Plotly.plot([CompScienceMeshes.wireframe(Γsrc), CompScienceMeshes.wireframe(Γtrg)])

##
MS = Maxwell3D.singlelayer(wavenumber=k)
MD = Maxwell3D.doublelayer(wavenumber=k)

Xsrc = raviartthomas(Γtrg)
Ytrg = buffachristiansen(Γsrc)

A = assemble(MD, Ytrg, Xsrc)

@views farblkasm = BEAST.blockassembler(
    MD,
    Ytrg,
    Xsrc,
    quadstrat=BEAST.defaultquadstrat(MD, Ytrg, Xsrc)
)

@views function farassembler(Z, tdata, sdata)
    @views store(v,m,n) = (Z[m,n] += v)
    farblkasm(tdata,sdata,store)
end

test_tree = create_tree(Ytrg.pos, FastBEAST.BoxTreeOptions(nmin=1000))
trial_tree = create_tree(Xsrc.pos, FastBEAST.BoxTreeOptions(nmin=1000))

lm = LazyMatrix(farassembler, FastBEAST.indices(test_tree), FastBEAST.indices(trial_tree), scalartype(MD));
fd = FastBEAST.FillDistance(Ytrg.pos)

# classic aca filldistance
@time U, V = aca(lm, rowpivstrat=fd, tol=1e-5);
@show size(U)
@show norm(U*V - A) / norm(A)

# adaptive aca
@time U, V = aca(lm, fd, maxrank=100, tol=1e-5);
@show size(U)
@show norm(U*V - A) / norm(A)

# classic aca max pivoting
@time U, V = aca(lm, maxrank=100, tol=1e-5);
@show size(U)
@show norm(U*V - A) / norm(A)
