struct ACAOptions{B, I, F}
    rowpivstrat::PivStrat
    columnpivstrat::PivStrat
    convcrit::ConvergenceCriterion
    maxrank::I
    tol::F
    svdrecompress::B
end


function ACAOptions(;
    rowpivstrat=MaxPivoting(),
    columnpivstrat=MaxPivoting(),
    convcrit=Standard(),
    maxrank=50,
    tol=1e-14,
    svdrecompress=false
)
    return ACAOptions(rowpivstrat, columnpivstrat, convcrit, maxrank, tol, svdrecompress)
end


function aca(
    M::LazyMatrix{I, K},
    am::ACAGlobalMemory{I, F, K};
    rowpivstrat::PivStrat=MaxPivoting(1),
    columnpivstrat::PivStrat=MaxPivoting(1),
    convcrit::ConvergenceCriterion=Standard(),
    maxrank=Int(round(length(M.τ)*length(M.σ)/(length(M.τ)+length(M.σ)))),
    tol=1e-14,
    svdrecompress=false
) where {I, F, K}
    
    clear!(am)  
    convcrit = initconvergence(M, convcrit)
    (maxrows, maxcolumns) = size(M)

    rowpivstrat, nextrow = firstpivot(rowpivstrat, M.τ)
    am.used_I[nextrow] = true

    @views M.μ(
        am.V[am.npivots:am.npivots, 1:maxcolumns], 
        M.τ[nextrow:nextrow],
        M.σ[1:maxcolumns]
    )

    @views nextcolumn = pivoting(
        columnpivstrat,
        abs.(am.V[am.npivots, 1:maxcolumns]),
        am.used_J[1:maxcolumns],
        convcrit
    )
    am.used_J[nextcolumn] = true

    dividor = am.V[am.npivots, nextcolumn]
    if dividor != 0
        @views am.V[am.npivots, 1:maxcolumns] ./= dividor
    end

    @views M.μ(
        am.U[1:maxrows, am.npivots:am.npivots], 
        M.τ[1:maxrows], 
        M.σ[nextcolumn:nextcolumn]
    )

    @views normU = norm(am.U[1:maxrows, am.npivots])
    @views normV = norm(am.V[am.npivots, 1:maxcolumns])
    
    normUV = normU*normV

    if isapprox(normU, 0.0) && isapprox(normV, 0.0)
        isconverged = true
    else
        isconverged, rowpivstrat, columnpivstrat = checkconvergence(
            normUV,
            maxrows,
            maxcolumns,
            am,
            rowpivstrat,
            columnpivstrat,
            convcrit,
            tol
        )
    end

    niter = 1
    while !isconverged && niter < maxrank
        niter += 1
        am.npivots += 1
        
        @views nextrow = pivoting(
            rowpivstrat,
            abs.(am.U[1:maxrows, am.npivots-1]),
            am.used_I[1:maxrows],
            convcrit
        )
        am.used_I[nextrow] = true

        normUV == 0.0 && (am.npivots -= 1)

        @views M.μ(
            am.V[am.npivots:am.npivots, 1:maxcolumns],
            M.τ[nextrow:nextrow],
            M.σ[1:maxcolumns]
        )

        for k = 1:(am.npivots-1)
            for kk=1:maxcolumns
                am.V[am.npivots, kk] -= am.U[nextrow, k]*am.V[k, kk]
            end
        end

        @views nextcolumn = pivoting(
            columnpivstrat,
            abs.(am.V[am.npivots, 1:maxcolumns]),
            am.used_J[1:maxcolumns],
            convcrit
        )

        dividor = am.V[am.npivots, nextcolumn]
        if dividor != 0
            @views am.V[am.npivots, 1:maxcolumns] ./= dividor
        end
        am.used_J[nextcolumn] = true
        
        @views M.μ(
            am.U[1:maxrows, am.npivots:am.npivots], 
            M.τ[1:maxrows],
            M.σ[nextcolumn:nextcolumn]
        )
        

        for k = 1:(am.npivots-1)
            for kk = 1:maxrows
                @views am.U[kk, am.npivots] -= am.U[kk, k]*am.V[k, nextcolumn]
            end
        end
        
        @views normU = norm(am.U[1:maxrows, am.npivots])
        @views normV = norm(am.V[am.npivots, 1:maxcolumns])

        normUV = normU*normV

        if isapprox(normU, 0.0) && isapprox(normV, 0.0) 
            am.npivots -= 1
            isconverged = true
        else
            isconverged, rowpivstrat, columnpivstrat = checkconvergence(
                normUV,
                maxrows,
                maxcolumns,
                am,
                rowpivstrat,
                columnpivstrat,
                convcrit,
                tol
            )
        end
    end

    if svdrecompress && am.npivots > 1
        @views Q,R = qr(am.U[1:maxrows,1:am.npivots])
        @views U,s,V = svd(R*am.V[1:am.npivots,1:maxcolumns])

        opt_r = length(s)
        for i in eachindex(s)
            if s[i] < tol*s[1]
                opt_r = i
                break
            end
        end

        A = (Q*U)[1:maxrows, 1:opt_r]
        B = (diagm(s)*V')[1:opt_r, 1:maxcolumns]

        return A, B
    else
        return am.U[1:maxrows, 1:am.npivots], am.V[1:am.npivots, 1:maxcolumns]
    end
end

"""
    aca(
        M::LazyMatrix{I, K};
        rowpivstrat::PivStrat=MaxPivoting(1),
        columnpivstrat::PivStrat=MaxPivoting(1),
        convcrit::ConvergenceCriterion=Standard(),
        tol::F=1e-14,
        maxrank::I=40,
        svdrecompress=false
    ) where {I, K}

Adaptive cross approximation that requires a `FastBEAST.LazyMatrix{I, K}` which is the 
constructor of the matrix. 

# Arguments
- `M::LazyMatrix{I, K}`: Constructor of the matrix.
- `am::ACAGlobalMemory{I, F, K}`: Memory struct of the ACA can be reused if the ACA is called several times.
- `rowpivstrat::PivStrat=MaxPivoting(1)`: Pivoting strategy for the rows.
- `columnpivstrat::PivStrat=MaxPivoting(1)`: Pivoting strategy for the columns.
- `convcrit::ConvergenceCriterion=Standard()`: Convergence criterion.
- `maxrank::I=40`: Maxium allowed rank in the ACA, equals the maximum allowed number of pivots.
- `tol::F=1e-14`: Tolerance used in the convergence criterion.
- `svdrecompress=false`: Recompresson of the approximation using and ACA.

# Example

"""
function aca(
        M::LazyMatrix{I, K};
        rowpivstrat::PivStrat=MaxPivoting(1),
        columnpivstrat::PivStrat=MaxPivoting(1),
        convcrit::ConvergenceCriterion=Standard(),
        tol::F=real(K)(1e-14),
        maxrank::I=40,
        svdrecompress=false
    ) where {I, F, K}

    return aca(
        M,
        allocate_aca_memory(K, size(M, 1), size(M, 2); maxrank=maxrank),
        rowpivstrat=rowpivstrat,
        columnpivstrat=columnpivstrat,
        convcrit=convcrit,
        tol=tol,
        maxrank=maxrank,
        svdrecompress=svdrecompress
    )
end