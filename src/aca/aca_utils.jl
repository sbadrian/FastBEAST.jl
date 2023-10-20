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

""" 
    function LazyMatrix(μ::Function, τ::Vector{I}, σ::Vector{I}, ::Type{F}) where {I, F}

# Arguments 
- `μ::Function`: 
- `τ::Vector{I}`: 
- `σ::Vector{I}`: 
- `::Type{F}`: 

"""
function LazyMatrix(μ::Function, τ::Vector{I}, σ::Vector{I}, ::Type{F}) where {I, F}
    
    return LazyMatrix{I, F}(μ, τ, σ)
end

@views function (A::LazyMatrix{K, F})(Z::S, I, J) where {K, F, S <: AbstractMatrix{F}}

    A.μ(view(Z, I, J), view(A.τ, I), view(A.σ, J))
end

mutable struct ACAGlobalMemory{I, F, K}
    Ic::I
    Jc::I
    U::Matrix{K}
    V::Matrix{K}
    used_I::Vector{Bool}
    used_J::Vector{Bool}
    normUV::F
end

maxrank(acamemory::ACAGlobalMemory) = size(acamemory.U, 2)

""" 
    function allocate_aca_memory(::Type{F}, maxrows, maxcolumns; maxrank=40) where {F}

Preallocation of sorage for the ACA.

# Arguments 
- `::Type{F}`: Type of matrix entries.
- `maxrows`: Number of rows.
- `maxcolumns`: Number of columns.

# Optional Arguments 
- `maxrank=40`: Maximum rank for the ACA. The ACA stops of more rows/columns are used for
the approximation.

"""
function allocate_aca_memory(::Type{F}, maxrows, maxcolumns; maxrank=40) where {F}

    U = zeros(F, maxrows, maxrank)
    V = zeros(F, maxrank, maxcolumns)
    used_I = zeros(Bool, maxrows)
    used_J = zeros(Bool, maxcolumns)
    return ACAGlobalMemory(1, 1, U, V, used_I, used_J, 0.0)
end