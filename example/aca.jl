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
    #println("nextrowindex: ", nextrowindex)

    acacolumnindices = Integer[]

    next_global_row =  matrix(rowindices[nextrowindex],colindices[:])

    nextcolumnindex, maxval = smartmaxlocal(next_global_row, acausedcolumnindices)
    acausedcolumnindices[nextcolumnindex] = true
    #println("nextcolumnindex: ", nextcolumnindex)

    push!(acacolumnindices, nextcolumnindex)

    V = next_global_row / next_global_row[nextcolumnindex]

    next_global_column = matrix(rowindices,colindices[nextcolumnindex])
    #println(size(next_global_column))
    U = next_global_column

    normUVlastupdate = norm(U)*norm(V)
    normUVsqared = normUVlastupdate^2
    i = 2
    while normUVlastupdate > sqrt(normUVsqared)*tol && 
            i <= length(rowindices) &&  i <= length(colindices)
        #println("Iteration: ", i)
        #println("Should we have stopped? ", normUVlastupdate > sqrt(normUVsqared)*tol ? "No" : "Yes")
        nextrowindex, maxval = smartmaxlocal(U[:,end], acausedrowindices)
        #println("nextrowindex: ", nextrowindex)
        if nextrowindex == -1
            error("Failed to find new row index: ", nextrowindex)
        end
        if isapprox(maxval, 0.0)
            println("Future V entry is close to zero. Abort.")
            return V, U
        end
        acausedrowindices[nextrowindex] = true

        next_global_row =  matrix(rowindices[nextrowindex],colindices[:])

        next_global_row -= (U[nextrowindex:nextrowindex, :]*V')'
        nextcolumnindex, maxval = smartmaxlocal(next_global_row, acausedcolumnindices)

        if (isapprox(next_global_row[nextcolumnindex],0.0))
            normUVlastupdate = 0.0
            println("Matrix seems to have exact rank: ", length(acarowindices))
        else
            push!(acarowindices, nextrowindex)

            V = hcat(V, next_global_row / next_global_row[nextcolumnindex])

            #println("nextcolumnindex: ", nextcolumnindex, " and ", maxval)
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

            next_global_column -= U*V[nextcolumnindex:nextcolumnindex, 1:end-1]'
            U = hcat(U,next_global_column)

            normUVlastupdate = norm(U[:,end])*norm(V[:,end])

            normUVsqared += normUVlastupdate^2
            for j = 1:(length(acacolumnindices))
                normUVsqared += 2*abs(dot(U[:,end], U[:,j])*dot(V[:,end], V[:,j]))
            end

            #println("normUVlastupdate: ", normUVlastupdate)
            #println("normUV: ", sqrt(normUVsqared))
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

V, U, acarowindices, acacolumnindices = aca_compression(fct, 1:N, 1:N)

@test U*V' ≈ A atol = 1e-14

##
Ns = 200
Nt = 100
A = rand(Nt,Ns)

fct(x,y) =  A[x,y]

U,S,V = svd(A)

S .= 0
S[1:15] = [10.0^(-i) for i = 1:15 ]

A = U*diagm(S)*V'

V, U, acarowindices, acacolumnindices = aca_compression(fct, 1:Nt, 1:Ns)

@test U*V' ≈ A atol = 1e-14

##
Ns = 100
Nt = 200
A = rand(Nt,Ns)

fct(x,y) =  A[x,y]

U,S,V = svd(A)

S .= 0
S[1:15] = [10.0^(-i) for i = 1:15 ]

A = U*diagm(S)*V'

V, U, acarowindices, acacolumnindices = aca_compression(fct, 1:Nt, 1:Ns)

@test U*V' ≈ A atol = 1e-14

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

@test U*V' ≈ A atol = 1e-14

##
Ns = 100
Nt = 200
A = rand(Nt,Ns)

B = zeros(2*Nt, 2*Ns)

U,S,V = svd(A)

S .= 0
S[1:15] = [10.0^(-i) for i = 1:15 ]

A = U*diagm(S)*V'

rowindices = [2*Nt - 2*i + 1 for i=1:Nt]
colindices = [2*Ns - 2*i + 1 for i=1:Ns]

for i = 1:Ns
    for j=1:Nt
        B[end-2*j+1,end-2*i+1] = A[j,i]
    end
end

fct(x,y) =  B[x,y]

V, U, acarowindices, acacolumnindices = aca_compression(fct, rowindices, colindices)

@test U*V' ≈ A atol = 1e-14
