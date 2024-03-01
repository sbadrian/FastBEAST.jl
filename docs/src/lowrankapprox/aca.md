# Adaptive Cross Aproximation
The adaptive cross approximation (ACA) is an adaptive rank revealing matrix factorization approach that computes the factorization of a low-rank matrix $\bm A^{m\times n}$ 

```math
    \bm A^{m\times n} \approx \widetilde{\bm A}^{m\times n} =  \bm U^{m\times r} \bm V^{r\times n}\,,
```
where ideally
```math
\lVert \bm A^{m\times n} - \widetilde{\bm{A}}^{m\times n} \rVert_\text{F} \leq \varepsilon \lVert \bm A^{m\times n}\rVert_\text{F}\,,
``` 
with the Forbenius norm $\lVert \cdot \rVert_\mathrm{F}$ and the $\varepsilon$ the tolerance of the approximation.

##### API
```@docs
FastBEAST.aca
```

##### Example
```julia
using FastBEAST
using StaticArrays
using LinearAlgebra

function OneoverRkernel(testpoint::SVector{3,T}, sourcepoint::SVector{3,T}) where T
    if isapprox(testpoint, sourcepoint, rtol=eps()*1e1)
        return 0.0
    else
        return 1.0 / (norm(testpoint - sourcepoint))
    end
end

function assembler(kernel, testpoints, sourcepoints)
    kernelmatrix = zeros(
        promote_type(eltype(testpoints[1]),eltype(sourcepoints[1])), 
        length(testpoints),
        length(sourcepoints)
    )

    for i in eachindex(testpoints)
        for j in eachindex(sourcepoints)
            kernelmatrix[i,j] = kernel(testpoints[i], sourcepoints[j])
        end
    end

    return kernelmatrix
end

function assembler(kernel, matrix, testpoints, sourcepoints)
    for i in eachindex(testpoints)
        for j in eachindex(sourcepoints)
            matrix[i,j] = kernel(testpoints[i], sourcepoints[j])
        end
    end
end

N=1000
spoints = [@SVector rand(3) for i = 1:N]
tpoints = 0.1*[@SVector rand(3) for i = 1:N] + [SVector(2.0, 0.0, 0.0) for i = 1:N]
@views OneoverRkernelassembler(matrix, tdata, sdata) = assembler(
    OneoverRkernel, matrix, tpoints[tdata], spoints[sdata]
)
lm = LazyMatrix(OneoverRkernelassembler, Vector(1:N), Vector(1:N), Float64)
U, V = aca(lm; tol=1e-4)

```

## Convergence Criteria
The computation of the true error of the factorization after the $k$th iteration is inefficient. Therefore, typically an approximation of the error is used. 
This package provides the following convergence criteria.

### Standard Convergence Criterion 
The standard convergence criterion used in the ACA checks after each iteration if 
```math
\lVert \bm u_k \rVert \lVert \bm v_k \rVert \leq \varepsilon \lVert \widetilde{\bm{A}}_k^{m\times n} \rVert_\text{F}\,, 
```
where $\lVert \widetilde{\bm{A}}^{m\times n} \rVert_\text{F}$ can be computed efficiently by
```math
    \lVert \widetilde{\bm{A}}_k^{m\times n} \rVert_\text{F}^2 = \lVert \widetilde{\bm{A}}_{k-1}^{m\times n} \rVert_\text{F}^2 + 2 \mathcal{R}\left\{ \sum_{j=1}^{k-1} (\bm u_j^\ast \bm u_k) (\bm v_j^\ast \bm v_k)\right\} + (|\bm u_k||\bm v_k|)^2
```
#### API
```@docs
FastBEAST.Standard
```

### Random Sampling Convergence Criterion 
The random sampling convergence criterion checks the mean true error of the approximation for a given subset of matrix entries in $\bm A^{m\times n}$ after $k$ iterations. Convergence is met if 
```math
\sqrt{\text{mean}(|\bm{e}_r^2|)mn} \leq \varepsilon \lVert \widetilde{\bm{A}}_k^{m\times n} \rVert_F\,,
```
where $\bm{e}_r$ contains the true error for the set of random samples after the $k$th iteration.

#### API
```@docs
FastBEAST.RandomSampling
FastBEAST.RandomSampling(::Type{K}; factor=real(K)(1.0), nsamples=0) where K
```

### [Combined Convergence Criterion](@id CCC) 
The combined convergence criterion checks both the standard and random sampling criterion resulting in 
```math
\text{max}(\lVert \bm u_k \rVert \lVert \bm v_k \rVert,\sqrt{\text{mean}(|\bm{e}_r^2|)mn}) \leq \varepsilon \lVert \widetilde{\bm{A}}_k^{m\times n} \rVert_F\,.
```
#### API
```@docs
FastBEAST.Combined
FastBEAST.Combined(::Type{K}; factor=real(K)(1.0), nsamples=0) where K
```

## Pivoting Strategies 
The rows and columns of $\bm A^{m \times n}$ used for the factorization are selected following a so called pivoting strategy.
This package provides the following pivoting strategies.


### [Standard Partial Pivoting](@id standardpiv)
The standard partial pivoting selects the next row or column by the maximum value in modulo of the last column or row. 

#### API
```@docs
FastBEAST.MaxPivoting
FastBEAST.MaxPivoting(;firstindex=1)
```

### Fill-Distance Pivoting
The fill-distance pivoting selects the rows (or the columns) based on geometrical considerations, while the corresponding columns (or rows) are selected by the standard partial pivoting.
The fill-distance requires for each row (or column) a corresponding node with a position which might be the vertex, edge, or face where a basis function is defined. 
For a matrix $A^{m\times n}$ we define the set $X$ comprising all nodes corresponding to the rows. The rows used for the factorization are then selected such that the fill-distance 
```math 
    h_{X_k, X} = \mathrm{sup}_{x\in X}\,\mathrm{dist}(x, X_k)
```
is minimized from step $k$ to step $k+1$, where $X_k \subseteq X$ contains all nodes that are already selected. 

#### API
```@docs
FastBEAST.FillDistance
FastBEAST.FillDistance(pos::Vector{SVector{3, F}}) where F <: Real
``` 

### [Modified Fill-Distance Pivoting](@id MFDPivoting)
The modified fill-distance approach is in principle a fast implementation of the fill-distance pivoting with small modifications. The pivoting is also based on the fill-distance 
```math 
    h_{X_k, X} = \mathrm{sup}_{x\in X}\,\mathrm{dist}(x, X_k)\,.
```
But contrary to the fill-distance pivoting, where a numerically expensive minimization has to be computed the pivots are selected following
```math 
    i_k = \mathrm{arg}\,\mathrm{max}_{x \in X}(\mathrm{dist}(x, X_{k-1}))\,.
```
The first pivot is selected as the basis-function closest to the barycenter of the set of nodes $X$. 

#### API
```@docs
FastBEAST.ModifiedFillDistance
FastBEAST.ModifiedFillDistance(pos::Vector{SVector{3, F}}) where F <: Real
``` 

#### MRF Pivoting
The MRF pivoting strategy is an adaptive approach comprising the [standard partial pivoting](@ref standardpiv), the [modified fill-distance pivoting](@ref MFDPivoting), and a random sampling pivoting, where the active strategy is selected based on which criterion of the [combined convergence criterion](@ref CCC) is met (details can be found in [[4]](@ref refs)). 

```@raw html
<div align="center">
<img src="../../assets/algorithmhz.svg" width="responsive-image"/>
</div>
<br/>
```

##### API
```@docs
FastBEAST.MRFPivoting
FastBEAST.MRFPivoting(pos::Vector{SVector{I, F}}) where {I, F}
``` 
