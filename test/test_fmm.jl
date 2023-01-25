using BEAST
using MKL
using CompScienceMeshes
using StaticArrays
using Test

function greensfct(qp::Matrix, S, mesh::Mesh)
    G = zeros(ComplexF64, length(qp) * length(qp[1,1]), length(qp) * length(qp[1,1]))
    α = S.alpha
    γ = S.gamma
    for nf in eachindex(mesh.faces)
        for (np, point) in enumerate(qp[1, nf])
            ind1 = (nf - 1) * length(qp[1,1])
            for nf2 in eachindex(mesh.faces)
                ind2 = (nf2 - 1) * length(qp[1,1])
                for (np2, point2) in enumerate(qp[1, nf2])
                    R = norm(point.point.cart - point2.point.cart)
                    if R == 0
                        G[ind1 + np, ind2 + np2] = 0
                    else
                        G[ind1 + np, ind2 + np2] = α*exp(-γ*R)/(4*π*R)
                    end
                end
            end
        end
    end
    return G
end

#fullinteractions
function generate_refmesh(;angle=180)
    vertices = [
        SVector(0.0, 1.0, 0.0),
        SVector(-0.5, 0.0, 0.0),
        SVector(+0.5, 0.0, 0.0),
        SVector(1.0, 1.0, 0.0),
        SVector(0.5, 2.0, 0.0),
        SVector(-0.5, 2.0, 0.0),
        SVector(-1.0, 1.0, 0.0)
    ]
    
    triangles = [
        SVector(1, 2, 3),
        SVector(1, 3, 4),
        SVector(1, 4, 5),
        SVector(1, 5, 6),
        SVector(1, 6, 7),
        SVector(1, 7, 2)
    ]
    return Mesh(vertices, triangles)
end

r = 10.0
λ = 20 * r
k = 2 * π / λ

refmesh = generate_refmesh()
X0 = lagrangecxd0(refmesh)
S = Helmholtz3D.hypersingular(; gamma=im*k)
S.alpha
S.gamma
points, qp = meshtopoints(X0, 1)

charges = ComplexF64.(rand(Float64, length(X0.fns)))

fmm = assemble_fmm(points, points, HelmholtzFMMOptions(ComplexF64(k)))
B = getBmatrix(qp, X0)
G = greensfct(qp, S, refmesh)

GBx = (fmm*(B*charges))[:,1]
Ax = transpose(B) * conj.(GBx[:,1])
Ax_true = transpose(B) * G * B * charges

println(@test norm(Ax - Ax_true) ≈ 0 atol=1e-15)
##
#compression

r = 10.0
λ = 20 * r
k = 2 * π / λ

refmesh = meshsphere(r, 1)

#patch functions
X0 = lagrangecxd0(refmesh)
S = Helmholtz3D.singlelayer(; gamma=im*k)
points, qp = meshtopoints(X0, 1)


charges = ComplexF64.(rand(Float64, length(X0.fns)))

fmm = assemble_fmm(points, points, HelmholtzFMMOptions(ComplexF64(k)))
B = getBmatrix(qp, X0)
G = greensfct(qp, S, refmesh)



GBx = (fmm*(B*charges))[:,1]
Ax = transpose(B) * conj.(GBx[:,1])
Ax_true = transpose(B) * G * B * charges

println(@test norm(Ax - Ax_true) ≈ 0 atol=1e-6)

#pyramid functions
X1 = lagrangec0d1(refmesh)
S = Helmholtz3D.singlelayer(; gamma=im*k)
points, qp = meshtopoints(X1, 1)


charges = ComplexF64.(rand(Float64, length(X1.fns)))

fmm = assemble_fmm(points, points, HelmholtzFMMOptions(ComplexF64(k)))
B = getBmatrix(qp, X1)
G = greensfct(qp, S, refmesh)



GBx = (fmm*(B*charges))[:,1]
Ax = transpose(B) * conj.(GBx[:,1])
Ax_true = transpose(B) * G * B * charges

println(@test norm(Ax - Ax_true) ≈ 0 atol=1e-6)

##
p = [0 0 0; 1 1 1;2 2 2]
charges = [1,1,1]

fmm = assemble_fmm(p, p, LaplaceFMMOptions())
fmm(charges)[1] 

