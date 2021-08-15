using LinearAlgebra
using Test

function aca_compression(matrix::Function, rowindices, colindices; tol=1e-14)

    function smartmaxlocal(roworcolumn, acausedindices)
        maxval = -1
        index = -1
        for i=1:length(roworcolumn)
            if !acausedindices[i]
                if abs(roworcolumn[i]) > maxval
                    maxval = abs(roworcolumn[i]) 
                    index = i
                end
            end
        end
        return index, maxval
    end

    acausedrowindices = zeros(Bool,length(rowindices))
    acausedcolumnindices = zeros(Bool,length(colindices))

    nextrowindex = 1
    acarowindices = [nextrowindex]
    acausedrowindices[nextrowindex] = true
    println("nextrowindex: ", nextrowindex)

    acacolumnindices = Integer[]

    next_global_row =  matrix(rowindices[nextrowindex],colindices[:])

    nextcolumnindex, maxval = smartmaxlocal(next_global_row, acausedcolumnindices)
    acausedcolumnindices[nextcolumnindex] = true
    println("nextcolumnindex: ", nextcolumnindex)

    push!(acacolumnindices, nextcolumnindex)

    V = next_global_row[rowindices] / next_global_row[nextcolumnindex]

    next_global_column = matrix(rowindices,colindices[nextcolumnindex])
    
    U = next_global_column[colindices]


    normUVlastupdate = norm(U)*norm(V)
    normUVsqared = normUVlastupdate^2
    i = 2
    while normUVlastupdate > sqrt(normUVsqared)*tol && i <= length(rowindices) &&  i <= length(colindices)
        println("Iteration: ", i)
        println("Should we have stopped? ", normUVlastupdate > sqrt(normUVsqared)*tol ? "No" : "Yes")
        nextrowindex, maxval = smartmaxlocal(U[:,end], acausedrowindices)
        println("nextrowindex: ", nextrowindex)
        if nextrowindex == -1
            error("Failed to find new row index: ", nextrowindex)
        end
        if isapprox(maxval, 0.0)
            println("Future V entry is close to zero. Abort.")
            return V, U
        end
        acausedrowindices[nextrowindex] = true

        next_global_row =  matrix(rowindices[nextrowindex],colindices[:])

        next_global_row[rowindices] -= (U[nextrowindex:nextrowindex, :]*V')'

        if (isapprox(next_global_row[rowindices[nextrowindex]],0.0))
            normUVlastupdate = 0.0
            println("Matrix seems to have exact rank: ", length(acarowindices))
        else
            push!(acarowindices, nextrowindex)

            nextcolumnindex, maxval = smartmaxlocal(next_global_row, acausedcolumnindices)

            V = hcat(V, next_global_row[rowindices] / next_global_row[rowindices[nextcolumnindex]])

            println("nextcolumnindex: ", nextcolumnindex, " and ", maxval)
            if nextcolumnindex == -1
                error("Failed to find new column index: ", nextcolumnindex)
            end
            if isapprox(maxval, 0.0)
                println("Future U entry is close to zero. Abort.")
                return V[:,1:end-1], U
            end
            acausedcolumnindices[nextcolumnindex] = true

            push!(acacolumnindices, nextcolumnindex)
            next_global_column = matrix(rowindices,colindices[nextcolumnindex])

            next_global_column[colindices] -= U*V[nextcolumnindex:nextcolumnindex, 1:end-1]'
            U = hcat(U,next_global_column[colindices])

            normUVlastupdate = norm(U[:,end])*norm(V[:,end])

            normUVsqared += normUVlastupdate^2
            for j = 1:(length(acacolumnindices))
                normUVsqared += 2*abs(dot(U[:,end], U[:,j])*dot(V[:,end], V[:,j]))
            end

            println("normUVlastupdate: ", normUVlastupdate)
            println("normUV: ", sqrt(normUVsqared))
        end
        i += 1
    end

    return V,U, acarowindices, acacolumnindices
end
##
N = 1000
A = rand(N,N)

fct(x,y) =  A[x,y]

U,S,V = svd(A)

S = [ i < 15 ? 10.0^(-i) : 0.0 for i = 1:N ]

A = U*diagm(S)*V'
##

V, U, acarowindices, acacolumnindices = aca_compression(fct, 1:N, 1:N)

@test U*V' ≈ A atol = 1e-4

##
a = [1.0 0.0 0.0 0.0 0.0]
b = [7.0 3.0 1.0 9.0 1.0]
A = a'*b

fct(x,y) =  A[x,y]

V, U = aca_compression(fct, 1:5, 1:5)

@test U*V' == A

##
a = [1.0 -2.0 6.0 4.0 5.0]
b = [7.0 3.0 1.0 9.0 1.0]
A = a'*b

fct(x,y) =  A[x,y]

V, U = aca_compression(fct, 1:5, 1:5)

@test U*V' == A

##
a1 = [1.0 -2.0 6.0 4.0 5.0]
b1 = [7.0 3.0 1.0 9.0 1.0]
a2 = [9.0 3.0 -6.0 2.0 3.0]
b2 = [5.0 4.0 3.0 1.0 -4.0]
#a3 = [10.0 2.0 -6.0 2.0 3.0]
#b3 = [-11.0 4.0 3.0 1.0 -4.0]
A = a1'*b1 + a2'*b2 #+ a3'*b3


fct(x,y) =  A[x,y]
V, U = aca_compression(fct, 1:5, 1:5)

@test U*V' ≈ A atol = 1e-4