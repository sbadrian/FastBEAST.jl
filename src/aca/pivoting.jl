# Standard partial pivoting
"""
    MaxPivoting{I} <: PivStrat

Struct for the standard partial pivoting of the ACA used for dispatching.

# Fields
- `firstindex::I`: Start index of the pivoting strategy.
"""
struct MaxPivoting{I} <: PivStrat
    firstindex::I
end

"""
    MaxPivoting(;firstindex=1)

Constructor for the standard partial pivoting of the ACA.

# Arguments
- `firstindex=1`: Start index of the pivoting strategy, default is the first index.
"""
function MaxPivoting(;firstindex=1)
    return MaxPivoting(firstindex)
end

function maxvalue(
    roworcolumn::Vector{K}, 
    usedidcs::Union{Vector{Bool}, SubArray{Bool, 1, Vector{Bool}, Tuple{UnitRange{Int}}, true}}
) where K
    
    idx = argmax(roworcolumn .* (.!usedidcs))
    (!usedidcs[idx]) && return idx 
    usedidcs[idx] && return argmin(usedidcs)
end

function firstpivot(pivstrat::MaxPivoting{Int}, globalidcs::Vector{Int})
    return pivstrat, pivstrat.firstindex
end

function pivoting(
    pivstrat::MaxPivoting,
    roworcolumn::Vector{K},
    usedidcs::Union{Vector{Bool}, SubArray{Bool, 1, Vector{Bool}, Tuple{UnitRange{Int}}, true}},
    convcrit::ConvergenceCriterion
) where K

    return maxvalue(roworcolumn, usedidcs)
end

# Fill-distance pivoting
abstract type FD <: PivStrat end

"""
    FillDistance{F <: Real} <: FD

Struct for the fill distance pivoting strategy of the ACA, used for dispatching.

# Fields
- `h::Vector{F}`: Distances of underlying nodes to the set of selected nodes.
- `pos::Vector{SVector{3, F}}`: Positions of the underlying nodes corresponding to the rows or columns.
"""
struct FillDistance{F <: Real} <: FD
    h::Vector{F}
    pos::Vector{SVector{3, F}}
end

"""
    FillDistance(pos::Vector{SVector{3, F}}) where F <: Real

Constructor for the fill distance pivoting strategy of the ACA.

# Arguments
- `pos::Vector{SVector{3, F}}`: Positions of the underlying nodes corresponding to the rows or columns.
"""
function FillDistance(pos::Vector{SVector{3, F}}) where F <: Real
    return FillDistance(zeros(F, length(pos)), pos)
end

function filldistance(
    fdmemory::FillDistance{F},
    usedidcs::Union{Vector{Bool}, SubArray{Bool, 1, Vector{Bool}, Tuple{UnitRange{Int}}, true}},
) where F <: Real

    pivots = Int[]
    maxval = maximum(fdmemory.h)

    for k in eachindex(fdmemory.h)
        newfd = 0.0
        for (ind, pos) in enumerate(fdmemory.pos)
            if fdmemory.h[ind] > norm(fdmemory.pos[k] - pos) 
                if newfd < norm(fdmemory.pos[k] - pos)
                    newfd = norm(fdmemory.pos[k] - pos)
                end
            else
                if newfd < fdmemory.h[ind]
                    newfd = fdmemory.h[ind]
                end
            end
        end
        if !usedidcs[k]
            if newfd < maxval
                maxval = newfd
                pivots = Int[k]
            elseif newfd == maxval 
                push!(pivots, k)
            end
        end
    end

    return pivots
end

"""
    ModifiedFillDistance{F <: Real} <: FD

Struct for the modified fill distance pivoting strategy of the ACA, used for dispatching.

# Fields
- `h::Vector{F}`: Distances of underlying nodes to the set of selected nodes.
- `pos::Vector{SVector{3, F}}`: Positions of the underlying nodes corresponding to the rows or columns.
"""
struct ModifiedFillDistance{F <: Real} <: FD
    h::Vector{F}
    pos::Vector{SVector{3, F}}
end

"""
    ModifiedFillDistance(pos::Vector{SVector{3, F}}) where F <: Real

Constructor for the modified fill distance pivoting strategy of the ACA.

# Arguments
- `pos::Vector{SVector{3, F}}`: Positions of the underlying nodes corresponding to the rows or columns.
"""
function ModifiedFillDistance(pos::Vector{SVector{3, F}}) where F <: Real
    return ModifiedFillDistance(zeros(F, length(pos)), pos)
end

function filldistance(
    fdmemory::ModifiedFillDistance{F},
    usedidcs::Union{Vector{Bool}, SubArray{Bool, 1, Vector{Bool}, Tuple{UnitRange{Int}}, true}},
) where F <: Real

    return [argmax(fdmemory.h)]
end

function update!(fdmemory::FD, pivotidx::Int)

    for k in eachindex(fdmemory.h)
        if fdmemory.h[k] > norm(fdmemory.pos[k] - fdmemory.pos[pivotidx])
            fdmemory.h[k] = norm(fdmemory.pos[k] - fdmemory.pos[pivotidx])
        end
    end
end

function firstpivot(pivstrat::FD, globalidcs::Vector{Int})

    localpos = pivstrat.pos[globalidcs]
    center = sum(localpos) / length(localpos)

    firstidcs = 0
    minimum = 0
    for (ind, pos) in enumerate(localpos)
        if ind == 1 || minimum > norm(pos - center)
            firstidcs = ind
            minimum = norm(pos-center)
        end
    end

    h = zeros(eltype(pivstrat.h), length(localpos))
    for i in eachindex(h)
        h[i] = norm(localpos[i] - localpos[firstidcs])
    end
    
    pivstrat isa FillDistance && return FillDistance(h, localpos), firstidcs
    pivstrat isa ModifiedFillDistance && return ModifiedFillDistance(h, localpos), firstidcs
    pivstrat isa MRFPivoting && return MRFPivoting(h, localpos, false, false, false), firstidcs
end

function pivoting(
    pivstrat::FD,
    roworcolumn::Vector{K},
    usedidcs::SubArray{Bool, 1, Vector{Bool}, Tuple{UnitRange{Int}}, true},
    convcrit::ConvergenceCriterion
) where {K}
    
    nextpivots::Vector{Int} = filldistance(pivstrat, usedidcs)
    nextpivot = nextpivots[1]
    @views length(nextpivots) > 1 && (
        nextpivot = nextpivots[argmax(roworcolumn[nextpivots])]
    )
    update!(pivstrat, nextpivot)
    
    return nextpivot
end

# MRFPivoting
"""
    MRFPivoting{I, F} <: FD

Struct for the MRF pivoting strategy of the ACA, used for dispatching.

# Fields
- `h::Vector{F}`: Distances of underlying nodes to the set of selected nodes.
- `pos::Vector{SVector{3, F}}`: Positions of the underlying nodes corresponding to the rows or columns.
- `sc::Bool`: Result of standard convergence criterion in the last iteration.
- `rc::Bool`: Result of random sampling convergence criterion in the last iteration.
- `fillstep::Bool`: True if a fill distance step has been done. 
"""
mutable struct MRFPivoting{I, F} <: FD
    h::Vector{F}
    pos::Vector{SVector{I, F}}
    sc::Bool
    rc::Bool
    fillstep::Bool
end

"""
    MRFPivoting(pos::Vector{SVector{I, F}}) where {I, F}

Constructor for the MRF pivoting strategy of the ACA.

# Arguments
- `pos::Vector{SVector{I, F}}`: Positions of the underlying nodes corresponding to the rows or columns.
"""
function MRFPivoting(pos::Vector{SVector{I, F}}) where {I, F}

    return MRFPivoting(zeros(F, length(pos)), pos, false, false, false)
end

function pivoting(
    pivstrat::MRFPivoting{3, F},
    roworcolumn::Vector{K},
    usedidcs::SubArray{Bool, 1, Vector{Bool}, Tuple{UnitRange{Int}}, true},
    # ToDo: Perhaps structs of Pivoting and Converegence must be declarded in aca_utils.jl
    convcrit::ConvergenceCriterion#Combined{I, F, K} 
) where {F, K}

    localind = 1
    if pivstrat.sc && pivstrat.rc && pivstrat.fillstep
        println("You should not be here.")
    elseif pivstrat.sc && pivstrat.rc && !pivstrat.fillstep
        pivstrat.fillstep = true
        localind = argmax(pivstrat.h)
    elseif pivstrat.sc
        localind = convcrit.indices[argmax(abs.(convcrit.rest)), 1]
    else
        localind = maxvalue(roworcolumn, usedidcs)
    end

    update!(pivstrat, localind)

    return localind
end