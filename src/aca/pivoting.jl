using Statistics
using LinearAlgebra

# Standard partial pivoting
struct MaxPivoting{I} <: PivStrat
    firstindex::I
end

function MaxPivoting()
    return MaxPivoting(1)
end

function maxvalue(
    roworcolumn::Vector{K}, 
    usedidcs::Union{Vector{Bool}, SubArray{Bool, 1, Vector{Bool}, Tuple{UnitRange{Int}}, true}}
) where K
    
    idx = argmax(roworcolumn .* (.!usedidcs))
    (!usedidcs[idx]) && return idx 
    usedidcs[idx] && return argmin(usedidcs)
end

function firstpivot(pivstrat::PivStrat, globalidcs::Vector{Int})
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
struct FillDistance{F <: Real} <: FD
    h::Vector{F}
    pos::Vector{SVector{3, F}}
end

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

struct ModifiedFillDistance{F <: Real} <: FD
    h::Vector{F}
    pos::Vector{SVector{3, F}}
end

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

""" 
    function firstpivot(pivstrat::FD, globalidcs::Vector{Int})

Returns first index of the pivoting strategy. For `FillDistance` this will be the 
basis function closest to the center of the distribution.

# Arguments 
- `pivstrat::FD`: Pivoting strategy.
- `globalidcs::Vector{Int}`: Indices corresponding to the matrix block, used to determine the
basis functions/positions used for pivoting.

"""
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
end


""" 
    function pivoting(
        pivstrat::FD,
        roworcolumn::Vector{K},
        usedidcs::SubArray{Bool, 1, Vector{Bool}, Tuple{UnitRange{Int}}, true},
        convcrit::ConvergenceCriterion
    ) where {K}

Returns next row or column used for approximation.

# Arguments 
- `pivstrat::FD`: Pivoting strategy. 
- `roworcolumn::Vector{K}`: Last row or column.
- `usedidcs::SubArray{Bool, 1, Vector{Bool}, Tuple{UnitRange{Int}}, true}`: Already used
indices. Rows/colums can be used only once.
- `convcrit::ConvergenceCriterion`: Convergence criterion used only in the case of MRFPivoting

"""
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

# EnforcedPivoting
mutable struct EnforcedPivoting{I, F} <: PivStrat
    firstindex::Int
    pos::Vector{SVector{I, F}}
    sc::Bool
    rc::Bool
    geostep::Bool
end

function EnforcedPivoting(pos::Vector{SVector{I, F}}; firstpivot=1) where {I, F}

    return EnforcedPivoting(firstpivot, pos, false, false, false)
end

function firstpivot(pivstrat::EnforcedPivoting{3, F}, globalidcs::Vector{Int}) where F
    return EnforcedPivoting(
        1, pivstrat.pos[globalidcs], false, false, false
    ), pivstrat.firstindex
end

function pivoting(
    pivstrat::EnforcedPivoting{3, F},
    roworcolumn::Vector{K},
    usedidcs::SubArray{Bool, 1, Vector{Bool}, Tuple{UnitRange{Int}}, true},
    # ToDo: Perhaps structs of Pivoting and Converegence must be declarded in aca_utils.jl
    convcrit::ConvergenceCriterion#Combined{I, F, K} 
) where {F, K}

    localind = 1
    if pivstrat.sc && pivstrat.rc && pivstrat.geostep
        println("You should not be here.")
    elseif pivstrat.sc && pivstrat.rc && !pivstrat.geostep
        #geostep
        pivstrat.geostep=true
        idcs = findall(x->x, usedidcs)
        pos = zeros(F, 3, length(idcs))
        mp = mean(pivstrat.pos[idcs])
        for (i, idx) in enumerate(idcs)
            pos[:, i] = pivstrat.pos[idx]-mp
        end
        u, _, _ = svd(pos)
        localind = argmin(usedidcs)
        maxval = 0.0
        for (lind, p) in enumerate(pivstrat.pos)
            val = abs(dot((p - mp), u[:, 3])) 
            if maxval < val && !usedidcs[lind]
                maxval = val 
                localind = lind
            end
        end
    elseif pivstrat.sc
        localind = convcrit.indices[argmax(abs.(convcrit.rest)), 1]
    else
        localind = maxvalue(roworcolumn, usedidcs)
    end

    return localind
end

