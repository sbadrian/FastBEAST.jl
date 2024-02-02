struct Standard <: ConvergenceCriterion end

""" 
    function initconvergence(
        M::LazyMatrix{I, K}, convcrit::Standard
    ) where {I, K} 

# Arguments 
- `M::FastBEAST.LazyMatrix{I, K}`: Assembler matrix used to compute rows and columns. 
- `convergcrit::Standard`: Convergence criterion used in the ACA, here used for dispaching. 

"""
function initconvergence(
    M::LazyMatrix{I, K}, convcrit::Standard
) where {I, K} 
    return convcrit
end

function convergence!(
    tol::F,
    normUV::F,
    am::ACAGlobalMemory{I, F, K},
    convcrit::Standard
) where {I, F <: Real, K}

    return normUV <= tol*sqrt(am.normUV²)
end

mutable struct RandomSampling{I, F <: Real, K} <: ConvergenceCriterion
    nsamples::I
    factor::F
    indices::Matrix{I}
    rest::Matrix{K}
end

function RandomSampling(::Type{K}; factor=1.0, nsamples=0) where K
    return RandomSampling(
        Int(ceil(nsamples*factor)),
        factor,
        zeros(Int, Int(ceil(nsamples*factor)), 2),
        zeros(K, Int(ceil(nsamples*factor)), 1)
    )
end

function (::RandomSampling{I, F, K})(
    ::Type{K}; factor=F(1.0), nsamples=I(0)
) where {I, F, K}
    return RandomSampling(
        I(ceil(nsamples*factor)),
        factor,
        zeros(I, I(ceil(nsamples*factor)), 2),
        zeros(K, I(ceil(nsamples*factor)), 1)
    )
end

function convergence!(
    tol::F,
    normUV::F,
    am::ACAGlobalMemory{I, F, K},
    convcrit::RandomSampling{I, F, K}
) where {I, F <: Real, K}

    # random sampling convergence
    for i in eachindex(convcrit.rest)
        @views convcrit.rest[i] -= 
            am.U[convcrit.indices[i, 1], am.npivots] * am.V[am.npivots, convcrit.indices[i, 2]] 
    end

    meanrest = sum(abs.(convcrit.rest).^2) / convcrit.nsamples

    return sqrt(meanrest*size(am.U, 1)*size(am.V, 2)) <= tol*sqrt(am.normUV²)
end

mutable struct Combined{I, F <: Real, K} <: ConvergenceCriterion
    nsamples::I
    factor::F
    indices::Matrix{I}
    rest::Matrix{K}
end

function Combined(::Type{K}; factor=1.0, nsamples=0) where K
    return Combined(
        Int(ceil(nsamples*factor)),
        factor,
        zeros(Int, Int(ceil(nsamples*factor)), 2),
        zeros(K, Int(ceil(nsamples*factor)), 1)
    )
end

function (::Combined{I, F, K})(::Type{K}; factor=1, nsamples=0) where {I, F, K}
    return Combined(
        Int(ceil(nsamples*factor)),
        factor,
        zeros(Int, Int(ceil(nsamples*factor)), 2),
        zeros(K, Int(ceil(nsamples*factor)), 1)
    )
end

function convergence!(
    tol::F,
    normUV::F,
    am::ACAGlobalMemory{I, F, K},
    convcrit::Combined{I, F, K}
) where {I, F <: Real, K}

    # random sampling convergence
    for i in eachindex(convcrit.rest)
        @views convcrit.rest[i] -= 
            am.U[convcrit.indices[i, 1], am.npivots] * am.V[am.npivots, convcrit.indices[i, 2]]
    end

    meanrest = sum(abs.(convcrit.rest).^2) / convcrit.nsamples

    return (sqrt(meanrest*size(am.U, 1)*size(am.V, 2)) <= tol*sqrt(am.normUV²) && 
        normUV <= tol*sqrt(am.normUV²))
end

""" 
    function initconvergence(
        M::LazyMatrix{I, K},
        convcrit::Union{RandomSampling{I, F, K}, Combined{I, F, K}},
    ) where {I, F <: Real, K}

Setup of the convergence criterion. Computation of the random samples, and allocation 
of the storage if not happened yet. 

# Arguments 
- `M::FastBEAST.LazyMatrix{I, K}`: Assembler matrix used to compute rows and columns.
- `convergcrit::Union{RandomSampling{I, K}, Combined{I, K}}`: Convergence criterion 
used in the ACA, here used for dispaching.

"""
function initconvergence(
    M::LazyMatrix{I, K},
    convcrit::Union{RandomSampling{I, F, K}, Combined{I, F, K}},
) where {I, F <: Real, K}

    convcrit.nsamples > length(M.τ)*length(M.σ) && println("Conv. oversampled!")

    convcrit.nsamples == 0 && (convcrit = convcrit(
        K, factor=convcrit.factor, nsamples=Int(ceil((size(M)[1] + size(M)[2])))
    ))

    convcrit.indices[1:convcrit.nsamples, 1] = rand(1:length(M.τ), convcrit.nsamples)
    convcrit.indices[1:convcrit.nsamples, 2] = rand(1:length(M.σ), convcrit.nsamples)
    for ind in eachindex(convcrit.rest)
        @views M.μ(
            convcrit.rest[ind:ind, 1:1], 
            M.τ[convcrit.indices[ind, 1]:convcrit.indices[ind, 1]],
            M.σ[convcrit.indices[ind, 2]:convcrit.indices[ind, 2]]
        )
    end

    return convcrit
end

""" 
    function checkconvergence(
        normUV::F,
        maxrows::I,
        maxcolumns::I,
        am::ACAGlobalMemory{I, F, K},
        rowpivstrat::PivStrat,
        columnpivstrat::PivStrat,
        convcrit::ConvergenceCriterion,
        tol::F
    ) where {I, F <: Real, K}

Checks if convergence in the ACA is reached.

# Arguments 
- `normUV::F`: Norm of last column times norm of las row.
- `maxrows::I`: Number of rows.
- `maxcolumns::I`: Number of columns.
- `am::ACAGlobalMemory{I, F, K}`: Preallocated memory used for the ACA. 
- `rowpivstrat::PivStrat`: Pivoting strategy for the rows.
- `columnpivstrat::PivStrat`: Pivoting strategy for the columns.
- `convergcrit::Standard`: Convergence criterion.
- `tol::F`: Tolerance of the ACA. 

"""
function checkconvergence(
    normUV::F,
    maxrows::I,
    maxcolumns::I,
    am::ACAGlobalMemory{I, F, K},
    rowpivstrat::PivStrat,
    columnpivstrat::PivStrat,
    convcrit::ConvergenceCriterion,
    tol::F
) where {I, F <: Real, K}

    if (normUV == 0) && (rowpivstrat != FastBEAST.MaxPivoting{I} || am.npivots == 1)
        rowpivstrat = FastBEAST.MaxPivoting()
        return false, rowpivstrat, columnpivstrat
    else
        am.normUV² += (normUV)^2
        for j = 1:(am.npivots-1)
            @views am.normUV² += 2*real.(
                dot(am.U[1:maxrows, am.npivots], am.U[1:maxrows, j]
            ) * dot(am.V[am.npivots, 1:maxcolumns], am.V[j, 1:maxcolumns]))
        end

        if normUV <= eps(real(K))*am.normUV²
            conv = convergence!(tol, normUV, am, convcrit)
            am.npivots -= 1
            return conv, rowpivstrat, columnpivstrat
        end

        return convergence!(tol, normUV, am, convcrit), rowpivstrat, columnpivstrat
    end
end

""" 
    function checkconvergence(
        normUV::F,
        maxrows::I,
        maxcolumns::I,
        am::ACAGlobalMemory{I, F, K},
        rowpivstrat::MRFPivoting,
        columnpivstrat::PivStrat,
        convcrit::ConvergenceCriterion,
        tol::F
    ) where {I, F <: Real, K}

Checks if convergence in the ACA is reached for MRFPivoting.

# Arguments 
- `normUV::F`: Norm of last column times norm of las row.
- `maxrows::I`: Number of rows.
- `maxcolumns::I`: Number of columns.
- `am::ACAGlobalMemory{I, F, K}`: Preallocated memory used for the ACA. 
- `rowpivstrat::MRFPivoting`: Pivoting strategy for the rows.
- `columnpivstrat::PivStrat`: Pivoting strategy for the columns.
- `convergcrit::Standard`: Convergence criterion.
- `tol::F`: Tolerance of the ACA. 

"""
function checkconvergence(
    normUV::F,
    maxrows::I,
    maxcolumns::I,
    am::ACAGlobalMemory{F, K},
    rowpivstrat::MRFPivoting,
    columnpivstrat::PivStrat,
    convcrit::ConvergenceCriterion,
    tol::F
) where {I, F <: Real, K}

    if (normUV == 0) && (am.npivots == 1)
        am.npivots -= 1 

        return false, rowpivstrat, columnpivstrat
    else
        am.normUV² += (normUV)^2
        for j = 1:(am.npivots-1)
            @views am.normUV² += 2*real.(
                dot(am.U[1:maxrows, am.npivots], am.U[1:maxrows, j]
            ) * dot(am.V[am.npivots, 1:maxcolumns], am.V[j, 1:maxcolumns]))
        end
        
         # random sampling convergence
         for i in eachindex(convcrit.rest)
            @views convcrit.rest[i] -= 
                am.U[convcrit.indices[i, 1], am.npivots] * am.V[am.npivots, convcrit.indices[i, 2]]
        end

        meanrest = sum(abs.(convcrit.rest).^2) / convcrit.nsamples
        lastupdate = rowpivstrat.sc && rowpivstrat.rc
        rowpivstrat.rc = sqrt(meanrest*size(am.U, 1)*size(am.V, 2)) <= tol*sqrt(am.normUV²)
        rowpivstrat.sc = normUV <= tol*sqrt(am.normUV²)
        conv = rowpivstrat.sc && rowpivstrat.rc && rowpivstrat.fillstep
        
        if lastupdate && conv
            am.npivots -= 1
        end
             
        return conv, rowpivstrat, columnpivstrat
    end
end 