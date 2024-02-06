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
    pivstrat isa MRFPivoting && return MRFPivoting(h, localpos, false, false, false), firstidcs
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

# MRFPivoting
mutable struct MRFPivoting{I, F} <: FD
    h::Vector{F}
    pos::Vector{SVector{I, F}}
    sc::Bool
    rc::Bool
    fillstep::Bool
end

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