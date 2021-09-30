function aca_compression(matrix::Function, rowindices, colindices; 
    tol=1e-14, T=ComplexF64, maxrank=40, svdrecompress=true, dblsupport=false)

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

    function update_approxxed!(times_approxxed, uv, estimated_error, tol)
        for (i, val) in enumerate(uv)
            if abs(val) > estimated_error*tol
                times_approxxed[i] += 1
            end
        end
    end

    numrows = length(rowindices)
    numcols = length(colindices)

    used_row_indices = zeros(Bool,numrows)
    used_column_indices = zeros(Bool,numcols)

    if dblsupport
        times_col_approxxed = zeros(Int, numcols)
        times_row_approxxed = zeros(Int, numrows)
    end

    row_indices_counter = 1
    column_indices_counter = 1


    maxrank = max(maxrank, length(rowindices), length(colindices))

    U = zeros(T, numrows, maxrank)
    V = zeros(T, maxrank, numcols)

    #acacolumnindices = Integer[]

    i = 1
    pivot_row = 1
    j = 1
    pivot_elem = 0.0
    pivot_col = -1

    # We might consider aborting earlier.
    # Maybe it is even reasonable to assume
    # that if a row is entirely zero, then
    # we are deadling with a zero matrix block
    while j <= numrows
        used_row_indices[pivot_row] = true

        @views matrix(V[row_indices_counter:row_indices_counter, :], 
                        rowindices[pivot_row:pivot_row],
                        colindices[:])

        @views pivot_col, pivot_elem = smartmaxlocal(V[row_indices_counter, :], used_column_indices)
        if pivot_col == -1 || isapprox(pivot_elem, 0.0)
      
            j += 1
            pivot_row += 1
            if j > numrows
                return U[:,1:1], V[1:1,:]
            end
        else
            break
        end
    end

    used_column_indices[pivot_col] = true


    #push!(acacolumnindices, pivot_col)

    @views V[row_indices_counter:row_indices_counter, :] /= V[row_indices_counter, pivot_col]

    @views matrix(U[:, column_indices_counter:column_indices_counter], 
                    rowindices, 
                    colindices[pivot_col:pivot_col])

    @views normUVlastupdate = norm(U[:, 1])*norm(V[1, :])
    normUVsqared = normUVlastupdate^2
   
    if dblsupport
        update_approxxed!(times_col_approxxed, 
            V[row_indices_counter:row_indices_counter, :],
            normUVlastupdate, tol)

        update_approxxed!(times_row_approxxed, 
            U[:, column_indices_counter:column_indices_counter],
            normUVlastupdate, tol)
    end

    isaltpivotstrategy = false

    if dblsupport
        current_minimum_row_approx = -1
    else
        current_minimum_row_approx = 1
    end

    while i <= numrows-1 &&  i <= numcols-1 && column_indices_counter < maxrank
        if normUVlastupdate < sqrt(normUVsqared)*tol
            if dblsupport && current_minimum_row_approx <= 0 # TODO: consider increasing 0
                current_minimum_row_approx = minimum(times_row_approxxed)

                if current_minimum_row_approx <= maximum(times_row_approxxed)
                    isaltpivotstrategy = true
                end
            else
                # At this point, we are finished
                break
            end
        end

        i += 1

        if isaltpivotstrategy == false #&& normUVlastupdate > sqrt(normUVsqared)*tol
            @views pivot_row, maxval = smartmaxlocal(U[:,column_indices_counter], used_row_indices)
        else
            # In the case that doublelayersupport is active
            pivot_row = -1
            for (i, val) in enumerate(used_row_indices)
                # Pick the first row that has not been sampled yet AND 
                # which according to the heuristic has not been updated
                if !val && times_row_approxxed[i] == current_minimum_row_approx
                    pivot_row = i
                    break
                end
            end
            normUVsqared = 0 # Reset the counter, we get a new block
            # We touched every row at least ones. Exit
            if pivot_row == -1
                break
            end
            isaltpivotstrategy = false
        end

        used_row_indices[pivot_row] = true

        row_indices_counter += 1
        @views matrix(V[row_indices_counter:row_indices_counter, :], 
                        rowindices[pivot_row:pivot_row],
                        colindices[:])

        @views V[row_indices_counter:row_indices_counter, :] -= U[pivot_row:pivot_row, 1:column_indices_counter]*V[1:(row_indices_counter-1), :]
        @views pivot_col, maxval = smartmaxlocal(V[row_indices_counter, :], used_column_indices)

        if pivot_col == -1
            println("i", i)
            println(used_column_indices)
            println("V[row_indices_counter, :]")
            println(V[row_indices_counter, :])
            println("U[pivot_row:pivot_row, 1:column_indices_counter]")
            println(U[pivot_row:pivot_row, 1:column_indices_counter])
            println("V[1, :]")
            println(V[1, :])
            println("pivot_elem, ", pivot_elem)
        end
        # TODO: comparing against zero: what atol should we use? 
        # PROPOSAL: put in relationship to maximum element so far
        if (isapprox(V[row_indices_counter, pivot_col],0.0))
            # Annhiliate last step
            V[row_indices_counter:row_indices_counter, :] .= 0.0
            row_indices_counter -= 1

            if dblsupport
                #maximum_row_approx = maximum(times_row_approxxed)
                current_minimum_row_approx = minimum(times_row_approxxed)

                if current_minimum_row_approx == 0
                    isaltpivotstrategy = true
                end
            end

            normUVlastupdate = 0.0
        else
            #push!(acarowindices, pivot_row)

            @views V[row_indices_counter:row_indices_counter, :] /= V[row_indices_counter, pivot_col]

            #if pivot_col == -1
            #    error("Failed to find new column index: ", pivot_col)
            #end

            #if isapprox(maxval, 0.0)
            #    println("Future U entry is close to zero. Abort.")
            #    return U[:,1:acacolumnindicescounter], V[1:acarowindicescounter-1,:]
            #end

            used_column_indices[pivot_col] = true

            #push!(acacolumnindices, pivot_col)

            column_indices_counter += 1
            @views matrix(U[:, column_indices_counter:column_indices_counter], 
                            rowindices,
                            colindices[pivot_col:pivot_col])

            @views U[:, column_indices_counter] -= U[:, 1:(column_indices_counter-1)]*V[1:(row_indices_counter-1),pivot_col:pivot_col]
            
            @views normUVlastupdate = norm(U[:,column_indices_counter])*norm(V[row_indices_counter,:])

            normUVsqared += normUVlastupdate^2
            for j = 1:(column_indices_counter-1)
                @views normUVsqared += 2*abs(dot(U[:,column_indices_counter], U[:,j])*dot(V[row_indices_counter,:], V[j,:]))
            end

            if dblsupport
                update_approxxed!(times_col_approxxed, 
                    V[row_indices_counter:row_indices_counter, :],
                    normUVlastupdate, tol)
        
                update_approxxed!(times_row_approxxed, 
                    U[:, column_indices_counter],
                    normUVlastupdate, tol)
            end

        end
        #println("i: ", i)
        #println("normUVlastupdate: ", normUVlastupdate)
        #println("sqrt(normUVsqared)*tol: ", sqrt(normUVsqared)*tol)
        #println("pivot_row: ", pivot_row)
    end

    if column_indices_counter == maxrank
        println("WARNING: aborted ACA after maximum allowed rank")
    end

    if svdrecompress && column_indices_counter > 1
        @views Q,R = qr(U[:,1:column_indices_counter])
        @views U,s,V = svd(R*V[1:row_indices_counter,:])

        opt_r = length(s)

        for i in eachindex(s)
            if s[i] < tol*s[1]
                opt_r = i
                break
            end
        end

        A = (Q*U)[:,1:opt_r]
        B = (diagm(s)*V')[1:opt_r,:]
        return A, B
    else
        return U[:,1:column_indices_counter], V[1:row_indices_counter,:]
    end    
end


function aca_compression(matrix::Function, testnode::BoxTreeNode, sourcenode::BoxTreeNode; 
                        tol=1e-14, T=ComplexF64, maxrank=40, svdrecompress=true, dblsupport=false)
    U, V = aca_compression(matrix, testnode.data, sourcenode.data,
                             tol=tol, T=T, maxrank=maxrank, svdrecompress=svdrecompress, dblsupport=dblsupport)

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
