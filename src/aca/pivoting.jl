using LinearAlgebra

abstract type PivStrat end

# max pivoting
struct MaxPivoting{I} <: PivStrat
    firstindex::I
end

function MaxPivoting()
    return MaxPivoting(1)
end

function firstindex(strat::MaxPivoting{I}, totalindices) where I
    return strat, strat.firstindex
end

function pivoting(
    strat::MaxPivoting{I},
    roworcolumn,
    acausedindices,
    totalindices
) where {I}
    
    if maximum(roworcolumn) != 0 
        return argmax(roworcolumn .* (.!acausedindices))
    else 
        return argmin(acausedindices)
    end
end

# fill distance pivoting
struct FillDistance{I, F} <: PivStrat
    loc::Vector{SVector{I, F}}
    dist::Vector{F}
end

function FillDistance(loc::Vector{SVector{I, F}}) where {I, F}

    return FillDistance(loc, zeros(F, length(loc)))
end

function firstindex(strat::FillDistance{I, F}, globalindices) where {I, F}
    
    nglobalindices = length(globalindices)
    distances = zeros(F, nglobalindices)
    dist = zeros(Float64, nglobalindices)
    
    firstlocalindex = 1
    
    for l = 1:nglobalindices
        dist[l] = norm(
            strat.loc[globalindices[l]] - strat.loc[globalindices[firstlocalindex]]
        )
    end
    
    maxdist = maximum(dist)

    for l = 2:nglobalindices
        for ll = 1:nglobalindices
            if maxdist < norm(strat.loc[globalindices[l]] - strat.loc[globalindices[ll]])
                distances[ll] = norm(
                    strat.loc[globalindices[l]] - strat.loc[globalindices[ll]]
                )
                break
            else
                distances[ll] = norm(
                    strat.loc[globalindices[l]] - strat.loc[globalindices[ll]]
                )
            end
        end
        if maximum(distances) < maximum(dist)
            maxdist = maximum(distances)
            firstlocalindex = l
            dist .= distances
        end
    end

    return FillDistance(strat.loc[globalindices], dist), firstlocalindex
end

function pivoting(
    strat::FillDistance{I, F},
    roworcolumn,
    acausedindices,
    totalindices
) where {I, F}

    if maximum(acausedindices) == 0
        firstind = firstindex(strat, totalindices)
        return firstind 
    else
        localind = argmax(strat.dist)

        for gind in eachindex(totalindices)
            if strat.dist[gind] > norm(strat.loc[gind] - strat.loc[localind])
                strat.dist[gind] = norm(strat.loc[gind] - strat.loc[localind])
            end
        end

        return localind
    end
end

# True fill distance pivoting
struct TrueFillDistance{I, F} <: PivStrat
    loc::Vector{SVector{I, F}}
    dist::Vector{F}
end

function TrueFillDistance(loc::Vector{SVector{I, F}}) where {I, F}

    return TrueFillDistance(loc, zeros(F, length(loc)))
end

function firstindex(strat::TrueFillDistance{I, F}, globalindices) where {I, F}
    
    nglobalindices = length(globalindices)
    distances = zeros(F, nglobalindices)
    dist = zeros(Float64, nglobalindices)
    
    firstlocalindex = 1
    
    for l = 1:nglobalindices
        dist[l] = norm(
            strat.loc[globalindices[l]] - strat.loc[globalindices[firstlocalindex]]
        )
    end
    
    maxdist = maximum(dist)

    for l = 2:nglobalindices
        for ll = 1:nglobalindices
            if maxdist < norm(strat.loc[globalindices[l]] - strat.loc[globalindices[ll]])
                distances[ll] = norm(
                    strat.loc[globalindices[l]] - strat.loc[globalindices[ll]]
                )
                break
            else
                distances[ll] = norm(
                    strat.loc[globalindices[l]] - strat.loc[globalindices[ll]]
                )
            end
        end
        if maximum(distances) < maximum(dist)
            maxdist = maximum(distances)
            firstlocalindex = l
            dist .= distances
        end
    end

    return TrueFillDistance(strat.loc[globalindices], dist), firstlocalindex
end

function pivoting(
    strat::TrueFillDistance{I, F},
    roworcolumn,
    acausedindices,
    totalindices
) where {I, F}

    if maximum(acausedindices) == 0
        firstind = firstindex(strat, totalindices)
        return firstind, maximum(strat.dist) 
    else
        nglobalindices = length(totalindices)
        maxdist = zeros(F, nglobalindices)
        distances = zeros(F, nglobalindices)
        
        for l = 1:nglobalindices
            for ll = 1:nglobalindices
                if strat.dist[ll] > norm(strat.loc[totalindices[l]] - strat.loc[totalindices[ll]])
                    distances[ll] = norm(strat.loc[totalindices[l]] - strat.loc[totalindices[ll]])
                else
                    distances[ll] = strat.dist[ll]
                end
            end
            maxdist[l] = maximum(distances)
        end

        ind = argmin(maxdist + maximum(maxdist) .* acausedindices) 

        for l = 1:nglobalindices
            if strat.dist[l] > norm(strat.loc[totalindices[l]] - strat.loc[totalindices[ind]])
                strat.dist[l] = norm(strat.loc[totalindices[l]] - strat.loc[totalindices[ind]])
            end 
        end

        return ind
    end
end