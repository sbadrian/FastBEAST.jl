using BEAST

struct ACAOptions{B, I, F}
    rowpivstrat::PivStrat
    columnpivstrat::PivStrat
    convergcrit::ConvergenceCriterion
    maxrank::I
    tol::F
    svdrecompress::B
end


function ACAOptions(;
    rowpivstrat=MaxPivoting(),
    columnpivstrat=MaxPivoting(),
    convergcrit=Standard(),
    maxrank=50,
    tol=1e-14,
    svdrecompress=false
)
    return ACAOptions(rowpivstrat, columnpivstrat, convergcrit, maxrank, tol, svdrecompress)
end

function aca(
    M::FastBEAST.LazyMatrix{I, K},
    am::ACAGlobalMemory{I, F, K};
    rowpivstrat=MaxPivoting(1),
    columnpivstrat=MaxPivoting(1),
    convergcrit=Standard(),
    tol=1e-14,
    svdrecompress=true
) where {I, F, K}

    initconvergence!(M, convergcrit)

    isconverged = false    

    (maxrows, maxcolumns) = size(M)

    rowpivstrat, nextrow = FastBEAST.firstindex(rowpivstrat, M.τ)
    am.used_I[nextrow] = true
    i = 1

    @views M.μ(
        am.V[am.Ic:am.Ic, 1:maxcolumns], 
        M.τ[nextrow:nextrow],
        M.σ[1:size(M,2)]
    )

    @views nextcolumn = FastBEAST.pivoting(
        columnpivstrat,
        abs.(am.V[am.Ic, 1:maxcolumns]),
        am.used_J[1:maxcolumns],
        M.σ
    )

    am.used_J[nextcolumn] = true

    dividor = am.V[am.Ic, nextcolumn]
    if dividor != 0
        @views am.V[am.Ic:am.Ic, 1:maxcolumns] ./= dividor
    end

    @views M.μ(
        am.U[1:maxrows, am.Jc:am.Jc], 
        M.τ[1:size(M, 1)], 
        M.σ[nextcolumn:nextcolumn]
    )

    @views normU = norm(am.U[1:maxrows, am.Jc])
    @views normV = norm(am.V[am.Ic, 1:maxcolumns])

    if isapprox(normU, 0.0) && isapprox(normV, 0.0)
        println("Matrix seems to have exact rank: ", am.Ic)
        isconverged = true
    else
        isconverged, rowpivstrat, columnpivstrat = checkconvergence(
            normU,
            normV,
            maxrows,
            maxcolumns,
            am,
            rowpivstrat,
            columnpivstrat,
            convergcrit,
            tol
        )
    end
    
    while !isconverged &&
        i <= length(M.τ)-1 &&
        i <= length(M.σ)-1 &&
        am.Jc < maxrank(am)

        i += 1
        
        @views nextrow = FastBEAST.pivoting(
            rowpivstrat,
            abs.(am.U[1:maxrows, am.Jc]),
            am.used_I[1:maxrows],
            M.τ
        )

        am.used_I[nextrow] = true

        # neglect zero update
        am.Ic != am.Jc && (am.Jc -= 1)

        am.Ic += 1

        @views M.μ(
            am.V[am.Ic:am.Ic, 1:maxcolumns],
            M.τ[nextrow:nextrow],
            M.σ[1:size(M, 2)]
        )

        @assert am.Jc == (am.Ic - 1)
        for k = 1:am.Jc
            for kk=1:maxcolumns
                am.V[am.Ic, kk] -= am.U[nextrow, k]*am.V[k, kk]
            end
        end

        @views nextcolumn = FastBEAST.pivoting(
            columnpivstrat,
            abs.(am.V[am.Ic, 1:maxcolumns]),
            am.used_J[1:maxcolumns],
            M.σ
        )

        dividor = am.V[am.Ic, nextcolumn]
        if dividor != 0
            @views am.V[am.Ic:am.Ic, 1:maxcolumns] ./= dividor
        end

        am.used_J[nextcolumn] = true
        am.Jc += 1
        
        @views M.μ(
            am.U[1:maxrows, am.Jc:am.Jc], 
            M.τ[1:size(M, 1)],
            M.σ[nextcolumn:nextcolumn]
        )
        
        @assert am.Jc == am.Ic
        for k = 1:(am.Jc-1)
            for kk = 1:maxrows
                am.U[kk, am.Jc] -= am.U[kk, k]*am.V[k, nextcolumn]
            end
        end

        @views normU = norm(am.U[1:maxrows, am.Jc])
        @views normV = norm(am.V[am.Ic, 1:maxcolumns])

        if isapprox(normU, 0.0) && isapprox(normV, 0.0)
            println("Matrix seems to have exact rank: ", am.Ic-1)
            am.Ic -= 1
            am.Jc -= 1
            isconverged = true
        else
            isconverged, rowpivstrat, columnpivstrat = checkconvergence(
                normU,
                normV,
                maxrows,
                maxcolumns,
                am,
                rowpivstrat,
                columnpivstrat,
                convergcrit,
                tol
            )
        end
    end

    if am.Jc == maxrank(am)
        println("WARNING: aborted ACA after maximum allowed rank")
    end

    if svdrecompress && am.Jc > 1
        @views Q,R = qr(am.U[1:maxrows,1:am.Jc])
        @views U,s,V = svd(R*am.V[1:am.Ic,1:maxcolumns])

        opt_r = length(s)
        for i in eachindex(s)
            if s[i] < tol*s[1]
                opt_r = i
                break
            end
        end

        A = (Q*U)[1:maxrows, 1:opt_r]
        B = (diagm(s)*V')[1:opt_r, 1:maxcolumns]

        am.U[1:maxrows, 1:am.Jc] .= 0.0
        am.V[1:am.Ic, 1:maxcolumns] .= 0.0
        am.used_J[1:maxcolumns] .= false
        am.used_I[1:maxrows] .= false
        am.Ic = 1
        am.Jc = 1
        am.normUV = 0.0

        return A, B
    else
        retU = am.U[1:maxrows, 1:am.Jc]
        retV = am.V[1:am.Ic, 1:maxcolumns]
        
        am.U[1:maxrows, 1:am.Jc] .= 0.0
        am.V[1:am.Ic, 1:maxcolumns] .= 0.0
        am.used_J[1:maxcolumns] .= false
        am.used_I[1:maxrows] .= false
        am.Ic = 1
        am.Jc = 1
        am.normUV = 0.0
        

        return retU, retV
    end
end

function aca(
    M::FastBEAST.LazyMatrix{I, F};
    rowpivstrat=FastBEAST.MaxPivoting(1),
    columnpivstrat=FastBEAST.MaxPivoting(1),
    convergcrit=Standard(),
    tol=1e-14,
    maxrank=40,
    svdrecompress=true
) where {I, F}

    return aca(
        M,
        allocate_aca_memory(F, size(M, 1), size(M, 2); maxrank=maxrank),
        rowpivstrat=rowpivstrat,
        columnpivstrat=columnpivstrat,
        convergcrit=convergcrit,
        tol=tol,
        svdrecompress=svdrecompress
    )
end