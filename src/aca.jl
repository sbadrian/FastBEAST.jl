function aca_compression(matrix::Function, rowindices, colindices; tol=1e-14, T=ComplexF64, maxrank=40, svdrecompress=true)

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
    #acarowindices = [nextrowindex]
    acausedrowindices[nextrowindex] = true

    U = zeros(T, length(rowindices), maxrank)
    V = zeros(T, maxrank, length(colindices))

    i = 1
    #acacolumnindices = Integer[]

    @views matrix(V[acarowindicescounter:acarowindicescounter, :], 
                    rowindices[nextrowindex:nextrowindex],
                    colindices[:])

    @views nextcolumnindex, maxval = smartmaxlocal(V[acarowindicescounter, :], acausedcolumnindices)
    acausedcolumnindices[nextcolumnindex] = true

    #push!(acacolumnindices, nextcolumnindex)

    @views V[acarowindicescounter:acarowindicescounter, :] /= V[acarowindicescounter, nextcolumnindex]

    @views matrix(U[:, acacolumnindicescounter:acacolumnindicescounter], 
                    rowindices, 
                    colindices[nextcolumnindex:nextcolumnindex])

    @views normUVlastupdate = norm(U[:, 1])*norm(V[1, :])
    normUVsqared = normUVlastupdate^2
   
    while normUVlastupdate > sqrt(normUVsqared)*tol && 
            i <= length(rowindices)-1 &&  i <= length(colindices)-1 && acacolumnindicescounter < maxrank

        i += 1

        @views nextrowindex, maxval = smartmaxlocal(U[:,acacolumnindicescounter], acausedrowindices)

        #if nextrowindex == -1
        #    error("Failed to find new row index: ", nextrowindex)
        #end

        #if isapprox(maxval, 0.0)
        #    println("Future V entry is close to zero. Abort.")
        #    return U[:,1:acacolumnindicescounter], V[1:acarowindicescounter, :]
        #end

        acausedrowindices[nextrowindex] = true

        acarowindicescounter += 1
        @views matrix(V[acarowindicescounter:acarowindicescounter, :], 
                        rowindices[nextrowindex:nextrowindex],
                        colindices[:])

        @views V[acarowindicescounter:acarowindicescounter, :] -= U[nextrowindex:nextrowindex, 1:acacolumnindicescounter]*V[1:(acarowindicescounter-1), :]
        @views nextcolumnindex, maxval = smartmaxlocal(V[acarowindicescounter, :], acausedcolumnindices)

        if (isapprox(V[acarowindicescounter, nextcolumnindex],0.0))
            normUVlastupdate = 0.0
            V[acarowindicescounter:acarowindicescounter, :] .= 0.0
            acarowindicescounter -= 1
            println("Matrix seems to have exact rank: ", acarowindicescounter)
        else
            #push!(acarowindices, nextrowindex)

            @views V[acarowindicescounter:acarowindicescounter, :] /= V[acarowindicescounter, nextcolumnindex]

            #if nextcolumnindex == -1
            #    error("Failed to find new column index: ", nextcolumnindex)
            #end

            #if isapprox(maxval, 0.0)
            #    println("Future U entry is close to zero. Abort.")
            #    return U[:,1:acacolumnindicescounter], V[1:acarowindicescounter-1,:]
            #end

            acausedcolumnindices[nextcolumnindex] = true

            #push!(acacolumnindices, nextcolumnindex)

            acacolumnindicescounter += 1
            @views matrix(U[:, acacolumnindicescounter:acacolumnindicescounter], 
                            rowindices,
                            colindices[nextcolumnindex:nextcolumnindex])

            @views U[:, acacolumnindicescounter] -= U[:, 1:(acacolumnindicescounter-1)]*V[1:(acarowindicescounter-1),nextcolumnindex:nextcolumnindex]
            
            @views normUVlastupdate = norm(U[:,acacolumnindicescounter])*norm(V[acarowindicescounter,:])

            normUVsqared += normUVlastupdate^2
            for j = 1:(acacolumnindicescounter-1)
                @views normUVsqared += 2*abs(dot(U[:,acacolumnindicescounter], U[:,j])*dot(V[acarowindicescounter,:], V[j,:]))
            end
        end
    end

    if acacolumnindicescounter == maxrank
        println("WARNING: aborted ACA after maximum allowed rank")
    end

    if svdrecompress && acacolumnindicescounter > 1
        @views Q,R = qr(U[:,1:acacolumnindicescounter])
        @views U,s,V = svd(R*V[1:acarowindicescounter,:])
        r′ = findfirst(x -> abs(x) < tol*s[1] ,s)
        r′ == 0 && (r′ = acacolumnindicescounter)
        A = (Q*U)[:,1:r′]
        B = (diagm(s)*V')[1:r′,:]
        return A, B
    else
        return U[:,1:acacolumnindicescounter], V[1:acarowindicescounter,:]
    end    
end


function aca_compression(matrix::Function, testnode::BoxTreeNode, sourcenode::BoxTreeNode; 
                        tol=1e-14, T=ComplexF64, maxrank=40, svdrecompress=true)
    U, V = aca_compression(matrix, testnode.data, sourcenode.data,
                             tol=tol, T=T, maxrank=maxrank, svdrecompress=svdrecompress)

    if false == true && size(U,2) >= 30
        println("Rank of compressed ACA is too large")
        println("Size of U: ", size(U))
        println("   Testnode box")
        println("    - Center: ", testnode.boundingbox.center)
        println("    - Halflength: ", testnode.boundingbox.halflength)
        println("    - Nodes: ", length(testnode.data))
        println("    - Level: ", testnode.level)
        println("   Sourcenode box")
        println("    - Center: ", sourcenode.boundingbox.center)
        println("    - Halflength: ", sourcenode.boundingbox.halflength)
        println("    - Nodes: ", length(sourcenode.data))
        println("    - Level: ", sourcenode.level)
        #UU, SS, VV = svd(U*V)
        #k = 1
        #while k < length(SS) && SS[k] > SS[1]*tol
        #    k += 1
        #end
        #println("   Actual rank: ", k)
        #error("emergency stop")
    end

    return U, V
end
