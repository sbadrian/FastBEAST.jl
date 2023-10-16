using BEAST

struct ACAOptions{B, I, F}
    rowpivstrat::PivStrat
    columnpivstrat::PivStrat
    maxrank::I
    tol::F
    svdrecompress::B
end

function ACAOptions(;
    rowpivstrat=MaxPivoting(),
    columnpivstrat=MaxPivoting(),
    maxrank=50,
    tol=1e-14,
    svdrecompress=false
)
    return ACAOptions(rowpivstrat, columnpivstrat, maxrank, tol, svdrecompress)
end

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

function allocate_aca_memory(::Type{F}, maxrows, maxcolumns; maxrank=40) where {F}

    U = zeros(F, maxrows, maxrank)
    V = zeros(F, maxrank, maxcolumns)
    used_I = zeros(Bool, maxrows)
    used_J = zeros(Bool, maxcolumns)
    return ACAGlobalMemory(1, 1, U, V, used_I, used_J, 0.0)
end

function checkconvergence(
    normU::F,
    normV::F,
    maxrows::I,
    maxcolumns::I,
    am::ACAGlobalMemory{I, F, K},
    rowpivstrat::FastBEAST.FillDistance{3, F},
    columnpivstrat::FastBEAST.MaxPivoting{I},
    tol::F,
) where {I, F, K}

    if normU == 0 || normV == 0
        #println("Alternatly structured matrixblock.")
        if am.Ic > 1
            am.Ic -= 1
            am.Jc -= 1
        end
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

function checkconvergence(
    normU::F,
    normV::F,
    maxrows::I,
    maxcolumns::I,
    am::ACAGlobalMemory{I, F, K},
    rowpivstrat::FastBEAST.MaxPivoting{I},
    columnpivstrat::FastBEAST.MaxPivoting{I},
    tol::F,
) where {I, F, K}

    if normU == 0 || normV == 0
        # println("Alternatly structured matrixblock.")
        if am.Ic > 1
            am.Ic -= 1
            am.Jc -= 1
        end

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

function aca(
    M::FastBEAST.LazyMatrix{I, K},
    am::ACAGlobalMemory{I, F, K};
    rowpivstrat=MaxPivoting(1),
    columnpivstrat=MaxPivoting(1),
    tol=1e-14,
    svdrecompress=true
) where {I, F, K}

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
            abs.(am.U[1:maxrows,am.Jc]),
            am.used_I[1:maxrows],
            M.τ
        )

        am.used_I[nextrow] = true

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
    tol=1e-14,
    maxrank=40,
    svdrecompress=true
) where {I, F}

    return aca(
        M,
        allocate_aca_memory(F, size(M, 1), size(M, 2); maxrank=maxrank),
        rowpivstrat=rowpivstrat,
        columnpivstrat=columnpivstrat,
        tol=tol,
        svdrecompress=svdrecompress
    )

end