function aca_compression(matrix::Function, rowindices, colindices; tol=1e-14, isdebug=false, maxrank=50)

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
    acarowindicescounter = 1
    acacolumnindicescounter = 1

    nextrowindex = 1
    acarowindices = [nextrowindex]
    acausedrowindices[nextrowindex] = true

    V = zeros(Float64, length(colindices), maxrank)
    U = zeros(Float64, length(rowindices), maxrank)

    i = 1
    acacolumnindices = Integer[]

    V[:, acarowindicescounter] =  matrix(rowindices[nextrowindex:nextrowindex],colindices[:])'

    nextcolumnindex, maxval = smartmaxlocal(V[:, acarowindicescounter], acausedcolumnindices)
    acausedcolumnindices[nextcolumnindex] = true
    #println("nextcolumnindex: ", nextcolumnindex)

    push!(acacolumnindices, nextcolumnindex)

    V[:, acarowindicescounter] /= V[nextcolumnindex, acarowindicescounter]

    U[:, acacolumnindicescounter] = matrix(rowindices,colindices[nextcolumnindex:nextcolumnindex])

    normUVlastupdate = norm(U[:, i])*norm(V[:, i])
    normUVsqared = normUVlastupdate^2
   
    while normUVlastupdate > sqrt(normUVsqared)*tol && 
            i <= length(rowindices)-1 &&  i <= length(colindices)-1 && acacolumnindicescounter < maxrank

        i += 1

        nextrowindex, maxval = smartmaxlocal(U[:,acacolumnindicescounter], acausedrowindices)

        if nextrowindex == -1
            error("Failed to find new row index: ", nextrowindex)
        end
        if isapprox(maxval, 0.0)
            println("Future V entry is close to zero. Abort.")
            return U[:,1:acacolumnindicescounter], V[:,1:acarowindicescounter]
        end

        acausedrowindices[nextrowindex] = true

        acarowindicescounter += 1
        V[:, acarowindicescounter] =  matrix(rowindices[nextrowindex:nextrowindex],colindices[:])'

        V[:, acarowindicescounter] -= (U[nextrowindex:nextrowindex, 1:acacolumnindicescounter]*V[:, 1:(acarowindicescounter-1)]')'
        nextcolumnindex, maxval = smartmaxlocal(V[:, acarowindicescounter], acausedcolumnindices)

        if (isapprox(V[nextcolumnindex, acarowindicescounter],0.0))
            normUVlastupdate = 0.0
            V[:, acarowindicescounter] .= 0.0
            acarowindicescounter -= 1
            println("Matrix seems to have exact rank: ", length(acarowindices))
        else
            push!(acarowindices, nextrowindex)

            V[:, acarowindicescounter] /= V[nextcolumnindex, acarowindicescounter]

            if nextcolumnindex == -1
                error("Failed to find new column index: ", nextcolumnindex)
            end

            if isapprox(maxval, 0.0)
                println("Future U entry is close to zero. Abort.")
                return U[:,1:acacolumnindicescounter], V[:,1:acarowindicescounter-1]
            end

            acausedcolumnindices[nextcolumnindex] = true

            push!(acacolumnindices, nextcolumnindex)

            acacolumnindicescounter += 1
            U[:, acacolumnindicescounter] = matrix(rowindices,colindices[nextcolumnindex:nextcolumnindex])

            U[:, acacolumnindicescounter] -= U[:, 1:(acacolumnindicescounter-1)]*V[nextcolumnindex:nextcolumnindex, 1:(acarowindicescounter-1)]'
            
            normUVlastupdate = norm(U[:,acacolumnindicescounter])*norm(V[:,acarowindicescounter])

            normUVsqared += normUVlastupdate^2
            for j = 1:acarowindicescounter
                normUVsqared += 2*abs(dot(U[:,end], U[:,j])*dot(V[:,end], V[:,j]))
            end
        end
    end

    if acacolumnindicescounter == maxrank
        println("WARNING: aborted ACA after maximum allowed rank")
    end

    return U[:,1:acacolumnindicescounter], V[:,1:acarowindicescounter]
end