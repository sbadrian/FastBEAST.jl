function aca(
    M::LazyMatrix{I, K},
    am::ACAGlobalMemory{K},
    rowpivstrat::FillDistance{3, F};
    columnpivstrat=MaxPivoting(1),
    tol=1e-4,
    svdrecompress=false
) where {I, F, K}
    Ic = 1
    Jc = 1
    normUVlastupdate = 0
    (maxrows, maxcolumns) = size(M)

    rowpivstrat, nextrow = firstindex(rowpivstrat, M.τ)
    am.used_I[nextrow] = true
    i = 1

    @views M.μ(
        am.V[Ic:Ic, 1:maxcolumns], 
        M.τ[nextrow:nextrow],
        M.σ[1:size(M,2)]
    )
    
    if norm(am.V[Ic:Ic, 1:maxcolumns]) == 0.0
        @views nextrow = pivoting(
            rowpivstrat,
            abs.(am.U[1:maxrows, Jc]),
            am.used_I[1:maxrows],
            M.τ
        )

        am.used_I[nextrow] = true
        @views M.μ(
        am.V[Ic:Ic, 1:maxcolumns], 
        M.τ[nextrow:nextrow],
        M.σ[1:size(M,2)]
    )
    end
    
    @views nextcolumn = pivoting(
        columnpivstrat,
        abs.(am.V[Ic, 1:maxcolumns]),
        am.used_J[1:maxcolumns],
        M.σ
    )
    am.used_J[nextcolumn] = true

    dividor = am.V[Ic, nextcolumn]
    if dividor != 0.0
        @views am.V[Ic:Ic, 1:maxcolumns] ./= dividor
    end
    
    @views M.μ(
        am.U[1:maxrows, Jc:Jc], 
        M.τ[1:size(M, 1)], 
        M.σ[nextcolumn:nextcolumn]
    )
    @views norm_U = norm(am.U[1:maxrows, 1])
    @views norm_V = norm(am.V[Ic,1:maxcolumns])
   
    if norm_U == 0.0 && norm_V != 0
        rowpivstrat = MaxPivoting()
        normUVlastupdate = norm_V
    elseif norm_U != 0.0 && norm_V == 0
        rowpivstrat = MaxPivoting()
        normUVlastupdate = norm_U
    else
        normUVlastupdate = norm_U * norm_V
    end
    
    normUVsqared = normUVlastupdate^2

    while normUVlastupdate > sqrt(normUVsqared)*tol && 
        i <= length(M.τ)-1 &&
        i <= length(M.σ)-1 &&
        Jc < maxrank(am)

        i += 1

        @views nextrow = pivoting(
            rowpivstrat,
            abs.(am.U[1:maxrows,Jc]),
            am.used_I[1:maxrows],
            M.τ
        )

        am.used_I[nextrow] = true

        Ic += 1
        @views M.μ(
            am.V[Ic:Ic, 1:maxcolumns],
            M.τ[nextrow:nextrow],
            M.σ[1:size(M, 2)]
        )

        @assert Jc == (Ic - 1)
        for k = 1:Jc
            for kk=1:maxcolumns
                am.V[Ic, kk] -= am.U[nextrow, k]*am.V[k, kk]
            end
        end

        @views nextcolumn = pivoting(
            columnpivstrat,
            abs.(am.V[Ic, 1:maxcolumns]),
            am.used_J[1:maxcolumns],
            M.σ
        )
        if isapprox(am.V[Ic, nextcolumn], 0.0) 
            if rowpivstrat isa MaxPivoting
                normUVlastupdate = 0.0
                am.V[Ic:Ic, 1:maxcolumns] .= 0.0
                Ic -= 1
                println("Matrix seems to have exact rank: ", Ic)
            else
                rowpivstrat = MaxPivoting()
                am.V[Ic:Ic, 1:maxcolumns] .= 0.0
                Ic -= 1
                i -= 1
            end
        else
            dividor = am.V[Ic, nextcolumn]

            if dividor != 0.0
                @views am.V[Ic:Ic, 1:maxcolumns] ./= dividor
            end

            am.used_J[nextcolumn] = true

            Jc += 1
            @views M.μ(
                am.U[1:maxrows, Jc:Jc], 
                M.τ[1:size(M, 1)],
                M.σ[nextcolumn:nextcolumn]
            )

            @assert Jc == Ic
            for k = 1:(Jc-1)
                for kk = 1:maxrows
                    am.U[kk, Jc] -= am.U[kk, k]*am.V[k, nextcolumn]
                end
            end

            @views normUVlastupdate = norm(am.U[1:maxrows, Jc]) * norm(am.V[Ic,1:maxcolumns])
            normUVsqared += normUVlastupdate^2
            for j = 1:(Jc-1)
                @views normUVsqared += 2*real(dot(am.U[1:maxrows,Jc], am.U[1:maxrows,j])*dot(am.V[Ic,1:maxcolumns], am.V[j,1:maxcolumns]))
            end
        end
    end

    if Jc == maxrank(am)
        println("WARNING: aborted ACA after maximum allowed rank")
    end

    if svdrecompress && Jc > 1
        @views Q,R = qr(am.U[1:maxrows,1:Jc])
        @views U,s,V = svd(R*am.V[1:Ic,1:maxcolumns])

        opt_r = length(s)
        for i in eachindex(s)
            if s[i] < tol*s[1]
                opt_r = i
                break
            end
        end

        A = (Q*U)[1:maxrows,1:opt_r]
        B = (diagm(s)*V')[1:opt_r,1:maxcolumns]

        am.U[1:maxrows, 1:Jc] .= 0.0
        am.V[1:Ic, 1:maxcolumns] .= 0.0
        am.used_J[1:maxcolumns] .= false
        am.used_I[1:maxrows] .= false

        return A, B
    else
        retU = am.U[1:maxrows,1:Jc]
        retV = am.V[1:Ic,1:maxcolumns]
        am.U[1:maxrows, 1:Jc] .= 0.0
        am.V[1:Ic, 1:maxcolumns] .= 0.0
        am.used_J[1:maxcolumns] .= false
        am.used_I[1:maxrows] .= false

        return retU, retV
    end    
end

function aca(
    M::LazyMatrix{I, K},
    rowpivstrat::FillDistance{3, F};
    columnpivstrat=MaxPivoting(1),
    tol=1e-4,
    maxrank=40,
    svdrecompress=false
) where {I, F, K}

    return aca(
        M,
        allocate_aca_memory(K, size(M, 1), size(M, 2); maxrank=maxrank),
        rowpivstrat,
        columnpivstrat=columnpivstrat,
        tol=tol,
        svdrecompress=svdrecompress
    )
end
