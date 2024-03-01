# FastBEAST

This package provides fast methods for boundary element simulations targeting [BEAST.jl](https://github.com/krcools/BEAST.jl). 

## Installation 
Installing FastBEAST is done by entering the package manager (enter `]` at the julia REPL) and issuing:

```
pkg> add https://github.com/sbadrian/FastBEAST.jl.git
```

## Overview
The following aspects are implemented (✔) and planned (⌛):

##### Available fast methods:
- ✔ H-matrix
- ✔ Kernel independent FMM based on [ExaFMM-t](https://github.com/exafmm/exafmm-t)
- ⌛ H2-matrix

##### Available low-rank compression techniques:
- ✔ Adaptive cross approximation 
- ⌛ Pseudo-skeleton approximation 
- ⌛ CUR matrix approximation 

## [References](@id refs)
The implementation is based on
- [1] Adelman, Ross, Nail A. Gumerov, and Ramani Duraiswami. “FMM/GPU-Accelerated Boundary Element Method for Computational Magnetics and Electrostatics.” IEEE Transactions on Magnetics 53, no. 12 (December 2017): 1–11. [https://doi.org/10.1109/TMAG.2017.2725951](https://doi.org/10.1109/TMAG.2017.2725951).
- [2] Bauer, M., M. Bebendorf, and B. Feist. “Kernel-Independent Adaptive Construction of $\mathcal {H}^2$-Matrix Approximations.” Numerische Mathematik 150, no. 1 (January 2022): 1–32. [https://doi.org/10.1007/s00211-021-01255-y](https://doi.org/10.1007/s00211-021-01255-y).
- [3] Heldring, Alexander, Eduard Ubeda, and Juan M. Rius. “Improving the Accuracy of the Adaptive Cross Approximation with a Convergence Criterion Based on Random Sampling.” IEEE Transactions on Antennas and Propagation 69, no. 1 (January 2021): 347–55. [https://doi.org/10.1109/TAP.2020.3010857](https://doi.org/10.1109/TAP.2020.3010857).
- [4] Tetzner, Joshua M., and Simon B. Adrian. “On the Adaptive Cross Approximation for the Magnetic Field Integral Equation.” Preprint. Preprints, January 26, 2024. [https://doi.org/10.36227/techrxiv.170630205.56494379/v1](https://doi.org/10.36227/techrxiv.170630205.56494379/v1).