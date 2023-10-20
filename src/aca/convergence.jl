abstract type ConvergenceCriterion end

struct Standard <: ConvergenceCriterion end

""" 
    function initconvergence!(
        M::FastBEAST.LazyMatrix{I, K}, 
        convergcrit::Standard
    ) where {I, K} end

# Arguments 
- `M::FastBEAST.LazyMatrix{I, K}`: Assembler matrix used to compute rows and columns. 
- `convergcrit::Standard`: Convergence criterion used in the ACA, here used for dispaching. 

"""
function initconvergence!(
    M::FastBEAST.LazyMatrix{I, K}, 
    convergcrit::Standard
) where {I, K} end

mutable struct RandomSampling{I, K} <: ConvergenceCriterion
    nsamples::I
    indices::Matrix{I}
    rest::Matrix{K}
end

""" 
    function RandomSampling(::Type{K}; nsamples=0) where K

Cunstructor for random-sampling convergence criterion. 

# Arguments 
- `::Type{K}`: Type of matrix entries.

# Optional Arguments 
- `nsamples=0`: Number of random samples used for the approximation. If not changed it will 
be number of rows + number of columns.

"""
function RandomSampling(::Type{K}; nsamples=0) where K
    return RandomSampling(nsamples, zeros(Int, nsamples, 2), zeros(K, nsamples, 1))
end

mutable struct Combined{I, K} <: ConvergenceCriterion
    nsamples::I
    indices::Matrix{I}
    rest::Matrix{K}
end

""" 
    function Combined(::Type{K}; nsamples=0) where K

Cunstructor for combined convergence criterion (random sampling and standard). 

# Arguments 
- `::Type{K}`: Type of matrix entries.

# Optional Arguments 
- `nsamples=0`: Number of random samples used for the approximation. If not changed it will 
be number of rows + number of columns.

"""
function Combined(::Type{K}; nsamples=0) where K
    return Combined(nsamples, zeros(Int, nsamples, 2), zeros(K, nsamples, 1))
end

""" 
    function initconvergence!(
        M::FastBEAST.LazyMatrix{I, K},
        convergcrit::Union{RandomSampling{I, K}, Combined{I, K}}
    ) where {I, K}

Setup of the convergence criterion. Computation of the random samples, and allocation 
of the storage if not happened yet. 

# Arguments 
- `M::FastBEAST.LazyMatrix{I, K}`: Assembler matrix used to compute rows and columns.
- `convergcrit::Union{RandomSampling{I, K}, Combined{I, K}}`: Convergence criterion 
used in the ACA, here used for dispaching.

"""
function initconvergence!(
    M::FastBEAST.LazyMatrix{I, K},
    convergcrit::Union{RandomSampling{I, K}, Combined{I, K}}
) where {I, K}

    if convergcrit.nsamples == 0
        convergcrit.nsamples = size(M)[1] + size(M)[2]
        convergcrit.indices = zeros(Int, convergcrit.nsamples, 2)
        convergcrit.rest = zeros(K, convergcrit.nsamples, 1)
    end

    convergcrit.indices[1:convergcrit.nsamples, 1] = 
        rand(1:length(M.τ), convergcrit.nsamples)
    convergcrit.indices[1:convergcrit.nsamples, 2] = 
        rand(1:length(M.σ), convergcrit.nsamples)
    
    for ind in eachindex(convergcrit.rest)
        @views M.μ(
            convergcrit.rest[ind:ind, 1:1], 
            M.τ[convergcrit.indices[ind, 1]:convergcrit.indices[ind, 1]],
            M.σ[convergcrit.indices[ind, 2]:convergcrit.indices[ind, 2]]
        )
    end
end

""" 
    function checkconvergence(
        normU::F,
        normV::F,
        maxrows::I,
        maxcolumns::I,
        am::ACAGlobalMemory{I, F, K},
        rowpivstrat::PivStrat,
        columnpivstrat::PivStrat,
        convergcrit::Standard,
        tol::F
    ) where {I, F, K}

Checks if convergence in the ACA is reached using the standard criterion.

# Arguments 
- `normU::F`: Norm of last column.
- `normV::F`: Norm of last row.
- `maxrows::I`: Number of rows.
- `maxcolumns::I`: Number of columns.
- `am::ACAGlobalMemory{I, F, K}`: Preallocated memory used for the ACA. 
- `rowpivstrat::PivStrat`: Pivoting strategy for the rows.
- `columnpivstrat::PivStrat`: Pivoting strategy for the rows.
- `convergcrit::Standard`: Convergence criterion.
- `tol::F`: Tolerance of the ACA. 

"""
function checkconvergence(
    normU::F,
    normV::F,
    maxrows::I,
    maxcolumns::I,
    am::ACAGlobalMemory{I, F, K},
    rowpivstrat::PivStrat,
    columnpivstrat::PivStrat,
    convergcrit::Standard,
    tol::F
) where {I, F, K}

    if normU == 0 || normV == 0
        am.Ic -= 1 
        rowpivstrat = FastBEAST.MaxPivoting()
        
        return false, rowpivstrat, columnpivstrat
    else
        am.normUV += (normU * normV)^2
        for j = 1:(am.Jc-1)
            @views am.normUV += 2*abs.(dot(am.U[1:maxrows, am.Jc], am.U[1:maxrows, j]) * dot(
                am.V[am.Ic, 1:maxcolumns],
                am.V[j, 1:maxcolumns])
            )
        end

        return normU*normV <= tol*sqrt(am.normUV), rowpivstrat, columnpivstrat
    end
end

""" 
    function checkconvergence(
        normU::F,
        normV::F,
        maxrows::I,
        maxcolumns::I,
        am::ACAGlobalMemory{I, F, K},
        rowpivstrat::PivStrat,
        columnpivstrat::PivStrat,
        convergcrit::Combined{I, K},
        tol::F,
    ) where {I, F, K}

Checks if convergence in the ACA is reached using the combined criterion.

# Arguments 
- `normU::F`: Norm of last column.
- `normV::F`: Norm of last row.
- `maxrows::I`: Number of rows.
- `maxcolumns::I`: Number of columns.
- `am::ACAGlobalMemory{I, F, K}`: Preallocated memory used for the ACA. 
- `rowpivstrat::PivStrat`: Pivoting strategy for the rows.
- `columnpivstrat::PivStrat`: Pivoting strategy for the rows.
- `convergcrit::Combined{I, K}`: Convergence criterion.
- `tol::F`: Tolerance of the ACA.

"""
function checkconvergence(
    normU::F,
    normV::F,
    maxrows::I,
    maxcolumns::I,
    am::ACAGlobalMemory{I, F, K},
    rowpivstrat::PivStrat,
    columnpivstrat::PivStrat,
    convergcrit::Combined{I, K},
    tol::F
) where {I, F, K}

    if normU == 0 || normV == 0
        rowpivstrat = FastBEAST.MaxPivoting()
        
        return false, rowpivstrat, columnpivstrat
    else
        # standard convergence
        am.normUV += (normU * normV)^2
        for j = 1:(am.Jc-1)
            @views am.normUV += 2*real(dot(am.U[1:maxrows, am.Jc], am.U[1:maxrows, j]) * dot(
                am.V[am.Ic, 1:maxcolumns],
                am.V[j, 1:maxcolumns])
            )
        end
        
        # random sampling convergence
        for i in eachindex(convergcrit.rest)
            @views convergcrit.rest[i] -= 
                am.U[convergcrit.indices[i, 1], am.Jc] * am.V[am.Ic, convergcrit.indices[i, 2]] 
        end

        meanrest = sum(abs.(convergcrit.rest).^2)/length(convergcrit.rest)
        conv = (sqrt(meanrest*maxrows*maxcolumns) <= tol*sqrt(am.normUV) && 
            normU*normV <= tol*sqrt(am.normUV))

        return conv, rowpivstrat, columnpivstrat
    end
end

""" 
    function checkconvergence(
        normU::F,
        normV::F,
        maxrows::I,
        maxcolumns::I,
        am::ACAGlobalMemory{I, F, K},
        rowpivstrat::PivStrat,
        columnpivstrat::PivStrat,
        convergcrit::RandomSampling{I, K},
        tol::F,
    ) where {I, F, K}

Checks if convergence in the ACA is reached using the random-sampling criterion.

# Arguments 
- `normU::F`: Norm of last column.
- `normV::F`: Norm of last row.
- `maxrows::I`: Number of rows.
- `maxcolumns::I`: Number of columns.
- `am::ACAGlobalMemory{I, F, K}`: Preallocated memory used for the ACA. 
- `rowpivstrat::PivStrat`: Pivoting strategy for the rows.
- `columnpivstrat::PivStrat`: Pivoting strategy for the rows.
- `convergcrit::RandomSampling{I, K}`: Convergence criterion.
- `tol::F`: Tolerance of the ACA.

"""
function checkconvergence(
    normU::F,
    normV::F,
    maxrows::I,
    maxcolumns::I,
    am::ACAGlobalMemory{I, F, K},
    rowpivstrat::PivStrat,
    columnpivstrat::PivStrat,
    convergcrit::RandomSampling{I, K},
    tol::F
) where {I, F, K}

    if normU == 0 || normV == 0
        rowpivstrat = FastBEAST.MaxPivoting()
        
        return false, rowpivstrat, columnpivstrat
    else
        # standard convergence
        am.normUV += (normU * normV)^2
        for j = 1:(am.Jc-1)
            @views am.normUV += 2*real(dot(am.U[1:maxrows, am.Jc], am.U[1:maxrows, j]) * dot(
                am.V[am.Ic, 1:maxcolumns],
                am.V[j, 1:maxcolumns])
            )
        end
        
        # random sampling convergence
        for i in eachindex(convergcrit.rest)
            @views convergcrit.rest[i] -= 
                am.U[convergcrit.indices[i, 1], am.Jc] * am.V[am.Ic, convergcrit.indices[i, 2]] 
        end

        meanrest = sum(abs.(convergcrit.rest).^2)/length(convergcrit.rest)
        conv = sqrt(meanrest*maxrows*maxcolumns) <= tol*sqrt(am.normUV)

        return conv, rowpivstrat, columnpivstrat
    end
end