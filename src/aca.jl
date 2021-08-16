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
            return U, V
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

    return U, V
end