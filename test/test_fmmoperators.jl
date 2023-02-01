using BEAST
using CompScienceMeshes
using ExaFMMt
using FastBEAST
using IterativeSolvers
using LinearAlgebra
using MKL
using StaticArrays
using Test

r = 10.0
λ = 20 * r
k = 2 * π / λ

sphere = meshsphere(r, 0.1 * r)
X0 = lagrangecxd0(sphere)
X1 = lagrangec0d1(sphere)

S = Helmholtz3D.singlelayer(; gamma=im * k)
D = Helmholtz3D.doublelayer(; gamma=im * k)
Dt = Helmholtz3D.doublelayer_transposed(; gamma=im * k)
N = Helmholtz3D.hypersingular(; gamma=im * k)

q = 100.0
ϵ = 1.0

# Interior problem
# Formulations from Sauter and Schwab, Boundary Element Methods(2011), Chapter 3.4.1.1
pos1 = SVector(r * 1.5, 0.0, 0.0)  # positioning of point charges
pos2 = SVector(-r * 1.5, 0.0, 0.0)

function Φ_inc(x)
    return q / (4 * π * ϵ) * (
        exp(-im * k * norm(x - pos1)) / (norm(x - pos1)) -
        exp(-im * k * norm(x - pos2)) / (norm(x - pos2))
    )
end

function ∂nΦ_inc(x)
    return -q / (4 * π * ϵ * r) * (
        (dot(
            x,
            (x - pos1) * exp(-im * k * norm(x - pos1)) / (norm(x - pos1)^2) *
            (im * k + 1 / (norm(x - pos1))),
        )) - (dot(
            x,
            (x - pos2) * exp(-im * k * norm(x - pos2)) / (norm(x - pos2)^2) *
            (im * k + 1 / (norm(x - pos2))),
        ))
    )
end

gD0 = assemble(ScalarTrace(Φ_inc), X0)
gD1 = assemble(ScalarTrace(Φ_inc), X1)
gN = assemble(ScalarTrace(∂nΦ_inc), X1)

G = assemble(BEAST.Identity(), X1, X1)
o = ones(numfunctions(X1))

# Interior Dirichlet problem
M_IDPSL = fmmassemble(
    S,
    X0,
    X0,
    threading=:multi,
    fmmoptions=HelmholtzFMMOptions(ComplexF64(k))
) # Single layer (SL)
M_IDPDL =  (-1 / 2 * assemble(BEAST.Identity(), X1, X1) + fmmassemble(
    D,
    X1,
    X1,
    threading=:multi,
    fmmoptions=HelmholtzFMMOptions(ComplexF64(k))
)) # Double layer (DL)
# Interior Neumann problem
# Neumann derivative from DL potential with deflected nullspace
M_INPDL = -fmmassemble(
    N,
    X1,
    X1,
    threading=:multi,
    nmin=20,
    fmmoptions=HelmholtzFMMOptions(ComplexF64(k))
) + G * o * o' * G
# Neumann derivative from SL potential with deflected nullspace
M_INPSL = (1 / 2 * assemble(BEAST.Identity(), X1, X1) + fmmassemble(
    Dt,
    X1,
    X1,
    threading=:multi,
    fmmoptions=HelmholtzFMMOptions(ComplexF64(k))
)) + G * o * o' * G 

ρ_IDPSL = IterativeSolvers.gmres(M_IDPSL, -gD0, verbose=true, reltol=1e-4, maxiter=200)
ρ_IDPDL = IterativeSolvers.gmres(M_IDPDL, gD1, verbose=true, reltol=1e-4, maxiter=200)
ρ_INPDL = IterativeSolvers.gmres(M_INPDL, gN, verbose=true, reltol=1e-4, maxiter=200)
ρ_INPSL = IterativeSolvers.gmres(M_INPSL, -gN, verbose=true, reltol=1e-4, maxiter=200)

pts = meshsphere(0.8 * r, 0.8 * 0.6 * r).vertices # sphere inside on which the potential and field are evaluated

pot_IDPSL = potential(HH3DSingleLayerNear(im * k), pts, ρ_IDPSL, X0; type=ComplexF64)
pot_IDPDL = potential(HH3DDoubleLayerNear(im * k), pts, ρ_IDPDL, X1; type=ComplexF64)

pot_INPSL = potential(HH3DSingleLayerNear(im * k), pts, ρ_INPSL, X1; type=ComplexF64)
pot_INPDL = potential(HH3DDoubleLayerNear(im * k), pts, ρ_INPDL, X1; type=ComplexF64)

# Total field inside should be zero
err_IDPSL_pot = norm(pot_IDPSL + Φ_inc.(pts)) / norm(Φ_inc.(pts))
err_IDPDL_pot = norm(pot_IDPDL + Φ_inc.(pts)) / norm(Φ_inc.(pts))

err_INPSL_pot = norm(pot_INPSL + Φ_inc.(pts)) / norm(Φ_inc.(pts))
err_INPDL_pot = norm(pot_INPDL + Φ_inc.(pts)) / norm(Φ_inc.(pts))

@test err_IDPSL_pot < 0.01
@test err_IDPDL_pot < 0.01
@test err_INPSL_pot < 0.01
@test err_INPDL_pot < 0.01
