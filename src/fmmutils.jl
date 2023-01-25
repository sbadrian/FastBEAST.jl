using BEAST
using SparseArrays
using LinearAlgebra

# fullrankblocks for corrections of fmm error
function getfullrankblocks(
    operator::BEAST.AbstractOperator,
    test_functions,
    trial_functions;
    threading=:single,
    nmin=10,
    quadstratcbk=BEAST.DoubleNumQStrat(1,1),
    quadstratfbk=BEAST.defaultquadstrat(operator, test_functions, trial_functions)
)

    @views farblkasm = BEAST.blockassembler(
        operator,
        test_functions,
        trial_functions,
        quadstrat=quadstratfbk
    )

    @views function farassembler(Z, tdata, sdata)
        @views store(v,m,n) = (Z[m,n] += v)
        farblkasm(tdata,sdata,store)
    end

    @views corblkasm = BEAST.blockassembler(
        operator,
        test_functions,
        trial_functions,
        quadstrat=quadstratcbk
    )

    @views function corassembler(Z, tdata, sdata)
        @views store(v,m,n) = (Z[m,n] += v)
        corblkasm(tdata,sdata,store)
    end

    MBF = FastBEAST.MatrixBlock{Int, scalartype(operator), Matrix{scalartype(operator)}}
    fullrankblocks = MBF[]
    fullrankblocks_perthread = Vector{MBF}[]
    correctionblocks = MBF[]
    correctionblocks_perthread = Vector{MBF}[]

    test_tree = create_tree(test_functions.pos, FastBEAST.BoxTreeOptions(nmin=nmin))
    trial_tree = create_tree(trial_functions.pos, FastBEAST.BoxTreeOptions(nmin=nmin))

    fullinteractions = []
    compressableinteractions = []

    FastBEAST.computerinteractions!(
        test_tree,
        trial_tree,
        fullinteractions,
        compressableinteractions
    )

    
    if threading == :single
        for fullinteraction in fullinteractions
            push!(
                fullrankblocks,
                FastBEAST.getfullmatrixview(
                    farassembler,
                    fullinteraction[1],
                    fullinteraction[2],
                    Int,
                    scalartype(operator)
                )
            )
            push!(
                correctionblocks,
                FastBEAST.getfullmatrixview(
                    corassembler,
                    fullinteraction[1],
                    fullinteraction[2],
                    Int,
                    scalartype(operator)
                )
            )
        end
    elseif threading == :multi
        for i in 1:Threads.nthreads()
            push!(fullrankblocks_perthread, MBF[])
            push!(correctionblocks_perthread, MBF[])
        end

        Threads.@threads for fullinteraction in fullinteractions
            push!(
                fullrankblocks_perthread[Threads.threadid()],
                FastBEAST.getfullmatrixview(
                    farassembler,
                    fullinteraction[1],
                    fullinteraction[2],
                    Int,
                    scalartype(operator)
                )
            )
            push!(
                correctionblocks_perthread[Threads.threadid()],
                FastBEAST.getfullmatrixview(
                    corassembler,
                    fullinteraction[1],
                    fullinteraction[2],
                    Int,
                    scalartype(operator)
                )
            )

        end

        for i in eachindex(fullrankblocks_perthread)
            append!(fullrankblocks, fullrankblocks_perthread[i])
        end
        for i in eachindex(correctionblocks_perthread)
            append!(correctionblocks, correctionblocks_perthread[i])
        end
    end

    return fullrankblocks, correctionblocks, fullinteractions
end

# mapping mesh to quadrature points
function meshtopoints(X::BEAST.LagrangeBasis, quadorder)

    test_elements, _, _ = assemblydata(X)
    tshapes = refspace(X)

    test_eval(x) = tshapes(x, Val{:withcurl})
    qp = quadpoints(test_eval,  test_elements, (quadorder,))
    points = zeros(Float64, length(qp) * length(qp[1,1]), 3)
    ind = 1
    for el in qp
        for i in el
            points[ind, 1] = i.point.cart[1]
            points[ind, 2] = i.point.cart[2]
            points[ind, 3] = i.point.cart[3]
            ind += 1
        end
    end
    
    return points, qp
end

# construction of B matrix, if Ax = b and A = transpose(B)GB
function getBmatrix(qp::Matrix, X::BEAST.LagrangeBasis)
    rfspace = refspace(X)
    _, tad, _ = assemblydata(X)
    len = length(qp) * length(qp[1,1]) * size(tad.data)[1] * size(tad.data)[2]
    rows = ones(Int, len)
    cols = ones(Int, len)
    vals = zeros(Float64, len)
    sind = 1

    for (ncell, cell) in enumerate(qp[1,:])
        ind = (ncell - 1) * length(cell)
        for (npoint, point) in enumerate(cell)
            val = rfspace(point.point)
            for localbasis in eachindex(val)
                for data in tad.data[:,localbasis,ncell]
                    if data[1] != 0 && ind + npoint != 0
                        rows[sind] = ind + npoint
                        cols[sind] = data[1]
                        vals[sind] = val[localbasis].value * point.weight * data[2]
                        sind += 1
                    end 
                end 
            end
        end
    end

    return dropzeros(sparse(rows, cols, vals))
end

function getBmatrix_curl(qp::Matrix, X::BEAST.LagrangeBasis, n)
    rfspace = refspace(X)
    _, tad, _ = assemblydata(X)
    len = length(qp) * length(qp[1,1]) * size(tad.data)[1] * size(tad.data)[2]
    rows = ones(Int, len)
    cols = ones(Int, len)
    vals = zeros(Float64, len)
    sind = 1

    for (ncell, cell) in enumerate(qp[1,:])
        ind = (ncell - 1) * length(cell)
        for (npoint, point) in enumerate(cell)
            val = rfspace(point.point)
            for localbasis in eachindex(val)
                for data in tad.data[:,localbasis,ncell]
                    if data[1] != 0 && ind + npoint != 0
                        rows[sind] = ind + npoint
                        cols[sind] = data[1]
                        vals[sind] = val[localbasis].curl[n] * point.weight * data[2]
                        sind += 1
                    end 
                end 
            end
        end
    end

    return dropzeros(sparse(rows, cols, vals))
end

# create sparse matrix from fullrankblocks
function fullmatrix(fullrankblocks)
    len = 0
    for fkb in fullrankblocks
        len += length(fkb.M)
    end
    
    rows = zeros(Int, len)
    cols = zeros(Int, len)
    vals = zeros(ComplexF64, len)
    ind = 1
    for fullrankblock in fullrankblocks
        for (i, row) in enumerate(fullrankblock.τ)
            for (j, col) in enumerate(fullrankblock.σ)
                rows[ind] = row
                cols[ind] = col
                vals[ind] = fullrankblock.M[i, j] 
                ind += 1
            end
        end
    end
    return sparse(rows, cols, vals)
end