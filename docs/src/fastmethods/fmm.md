# Fast Multipole Method

The fast multipole method allows to accelerate certain matrix-vector products, originally introduced for N-body problems arising in stellar and molecular dynamics. A matrix-vector product

$$\bm{A} \bm{x}  = \bm{y}$$

can be computed with the FMM to any specific accuracy $\varepsilon$ in $\mathcal{O}(N)$ in storage and time.
For the boundary element method (BEM) several implementations of the FMM exist which, in general, are kernel dependent. 

Adelmann et al. propose in [[1]](@ref refs) a method using FMM codes for monopole and dipole sources unchanged to evaluate integral expressions in the BEM, which allows to use existing highly optimized implementations of the FMM such as [ExaFMM](https://github.com/exafmm/exafmm-t). The approach of Adelmann is called the error correction factor matrix method (ECFMM). 

## Error Correction Factor Matrix Method
The ECFMM approximates the BEM integrals by a quadrature and treats the quadrature points as monopole and dipole sources, which can be evaluated by the FMM. Approximation errors of the quadrature are corrected during a correction factor step.
The matrix-vector product becomes

$$\bm{A} \bm{x} \approx \bm{P}_2^T(\bm{G}-\bm{C})\bm{P_1}\bm{x} + \bm{S}\bm{x}\approx y\,,$$

where $\bm{S}$ describes all not well-separated interactions, $\bm{G}$ resembles the FMM, $\bm{C}$ corrects the quadrature errors and $\bm{P}_{1, 2}$ describes the charge for each monopole or dipole source comprising the weight of the quadrature point and the action the test and trial functions.
Detailed explanations can be found in [FMM/GPU-Accelerated Boundary Element Method for Computational Magnetics and Electrostatics.](https://doi.org/10.1109/TMAG.2017.2725951)

## Example

```julia
using CompScienceMeshes
using FastBEAST

Γ = meshsphere(1.0, 0.1)

𝓣 = Maxwell3D.singlelayer(wavenumber=k)
X = raviartthomas(Γ)

T = fmmassemble(𝓣, X, X)
```