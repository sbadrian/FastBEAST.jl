abstract type ConvergenceCriterion end
abstract type PivStrat end

struct LazyMatrix{I, F} <: AbstractMatrix{F}
    μ::Function
    τ::Vector{I}
    σ::Vector{I}
end

Base.size(A::LazyMatrix) = (length(A.τ), length(A.σ))

function Base.getindex(
    A::T,
    I,
    J
) where {K, F, T <: LazyMatrix{K, F}}

    Z = zeros(F, length(I), length(J))
    A.μ(Z, view(A.τ, I), view(A.σ, J))
    return Z
end

function LazyMatrix(μ::Function, τ::Vector{I}, σ::Vector{I}, ::Type{F}) where {I, F}
    
    return LazyMatrix{I, F}(μ, τ, σ)
end

@views function (A::LazyMatrix{K, F})(Z::S, I, J) where {K, F, S <: AbstractMatrix{F}}

    A.μ(view(Z, I, J), view(A.τ, I), view(A.σ, J))
end

mutable struct ACAGlobalMemory{I, F <: Real, K}
    U::Matrix{K}
    V::Matrix{K}
    used_I::Vector{Bool}
    used_J::Vector{Bool}
    normUV²::F
    npivots::I
end

function clear!(am::ACAGlobalMemory{I, F, K}) where {I, F <: Real, K}
    am.U .= K(0.0)
    am.V .= K(0.0)
    am.used_I .= false
    am.used_J .= false
    am.normUV² = F(0.0)
    am.npivots = I(1)
end

maxrank(acamemory::ACAGlobalMemory) = size(acamemory.U, 2)

""" 
    function allocate_aca_memory(
        ::Type{K}, maxrows::I, maxcolumns::I; maxrank=40
    ) where {I, K}

Preallocation of sorage for the ACA.

# Arguments 
- `::Type{K}`: Type of matrix entries.
- `maxrows::I`: Number of rows.
- `maxcolumns::I`: Number of columns.

# Optional Arguments 
- `maxrank=40`: Maximum rank for the ACA. The ACA stops of more rows/columns are used for
the approximation.

"""
function allocate_aca_memory(
    ::Type{K}, maxrows::I, maxcolumns::I; maxrank=40
) where {I, K}
    U = zeros(K, maxrows, maxrank)
    V = zeros(K, maxrank, maxcolumns)
    used_I = zeros(Bool, maxrows)
    used_J = zeros(Bool, maxcolumns)

    return ACAGlobalMemory(U, V, used_I, used_J, real(K)(0.0), I(1))
end