# $\mathcal{H}$-Matrix

The $\mathcal{H}$-Matrix is form of storing boundary element matrices blockwise differentiating compressed low-rank and full-rank matrices.
The system matrix in the boundary element method (BEM) is in general dense but has blocks corresponding to well-separated clusters of test- and trial-functions which are low rank and can be compressed. 
 
Using this method the complexity of the computation an approximation with any specific accuracy $\varepsilon$ of a boundary element matrix is $\mathcal{O} (N\log N)$ in time and storage, as well for computing a matrix-vector product.

An overview of the low-rank approximation techniques available in this package can be found [here]()

## Example

```julia
using BEAST
using FastBEAST
using CompScienceMeshes

Γ = meshsphere(1.0, 0.1)

𝓣 = Maxwell3D.singlelayer(wavenumber=k)
X = raviartthomas(Γ)

T = hassemble(𝓣, X, X)
```