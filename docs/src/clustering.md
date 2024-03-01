# Clustering Strategies
This package provides an interface for the [ClusterTrees.jl](https://github.com/krcools/ClusterTrees.jl) package for preprocessing the geometry and detecting admissible clusters. Admissability criteria are defined for the following clustering algorithms.

## Box Tree 
The box tree is a classical octree for 3D and an quadtree for 2D geometries. 
The admissibility of the clusters is determined by
```math
\begin{equation*}
    \begin{aligned}
    \frac{\sqrt{3}}{2}(l_{X_\text{s}}+l_{X_\text{t}}) &\leq \eta \text{dist}_{\text{c}}(X_\text{s}, X_\text{t})\,, 
    \end{aligned}
\end{equation*}
```
where $\text{dist}_\text{c}(\cdot, \cdot)$ donates the euclidean distance between the center of the two boxes and $l_{X}$ donates the edge length of the box associated with the cluster $X$.
The parameter $\eta$ might be adjusted to the fast method, but should be $\eta \leq 1$. 

### Example
 ```julia
using ClusterTrees
using FastBEAST
using StaticArrays

N = 1000
nodes = [@SVector rand(Float64, 3) for i in 1:N]

tree = create_tree(nodes)

nears = SVector{2}[]
fars = SVector{2}[]
computerinteractions!(tree, tree, nears, fars)
```

## K-Means Clustering Tree
The K-Means algorithm is a clustering strategy for n-dimensional spaces, in which each cluster is represented by its barycenter and radius. 
The admissibility of clusters is determined by 
```math
\begin{equation*}
    \begin{aligned}
    2\text{max}(\text{rad}(X_\text{t}), \text{rad}(X_\text{s})) &\leq η\, \text{dist}_{\text{bc}}(X_\text{t}, X_\text{s})\\
    \text{dist}_{\text{bc}}(X, Y) &= \text{d}(\text{bc}(X), \text{bc}(Y)) + \text{rad}(X) + \text{rad}(Y)\,,
    \end{aligned}
\end{equation*}
```
where $\text{d}(\cdot,\cdot)$ donates the euclidean distance, $\text{rad}(X) = \text{sup}(\text{d}(\text{bc}(X), x)| x \in X)\,,$ and $\text{bc}(\cdot)$ the barycenter of the cluster defined in the K-Means algorithm.
The parameter $\eta$ might be adjusted to the fast method, but should be $\eta \leq 1$. 

### Example
```julia
using ClusterTrees
using FastBEAST
using StaticArrays

N = 1000
nodes = [@SVector rand(Float64, 3) for i in 1:N]

tree = create_tree(nodes, KMeansTreeOptions())

nears = SVector{2}[]
fars = SVector{2}[]
computerinteractions!(tree, tree, nears, fars)
```