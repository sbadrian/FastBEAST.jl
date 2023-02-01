using ExaFMMt
using BEAST
using LinearAlgebra
using LinearMaps
using SparseArrays
using StaticArrays

function assemble_fmm(
    spoints::Matrix{F},
    tpoints::Matrix{F};
    options=LaplaceFMMOptions() 
) where F <: Real
    
    A = setup(spoints, tpoints, options)
    return A

end

function assemble_fmm(
    spoints::Matrix{F},
    tpoints::Matrix{F},
    options::ExaFMMt.FMMOptions 
) where F <: Real

    A = setup(spoints, tpoints, options)
    return A

end

# fullrankblocks for corrections of fmm error
function getfullrankblocks(
    operator::BEAST.AbstractOperator,
    test_functions::BEAST.Space,
    trial_functions::BEAST.Space;
    threading=:single,
    nmin=10,
    quadstratcbk=BEAST.DoubleNumQStrat(1,1),
    quadstratfbk=BEAST.defaultquadstrat(operator, test_functions, trial_functions)
)

    @views blkasm = BEAST.blockassembler(operator, test_functions, trial_functions)

    @views function assembler(Z, tdata, sdata)
        @views store(v,m,n) = (Z[m,n] += v)
        blkasm(tdata,sdata,store)
    end

    @views corblkasm = BEAST.blockassembler(
        operator,
        test_functions,
        trial_functions,
        quadstrat=quadstratcbk
    )

    @views function corassembler(Z, tdata, sdata)
        @views store(v,m,n) = (Z[m,n] += v)
        corblkasm(tdata, sdata, store)
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
                    assembler,
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
                    assembler,
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
function meshtopoints(X::BEAST.Space, quadorder)

    test_elements, _, _ = assemblydata(X)
    tshapes = refspace(X)

    test_eval(x) = tshapes(x)
    qp = BEAST.quadpoints(test_eval,  test_elements, (quadorder,))
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
function getBmatrix(op::BEAST.Helmholtz3DOp, qp::Matrix, X::BEAST.Space)
    rfspace = refspace(X)
    _, tad, _ = assemblydata(X)
    len = length(qp) * length(qp[1,1]) * size(tad.data)[1] * size(tad.data)[2]

    rc = ones(Int, len, 2)
    vals = zeros(Float64, len)
    sind = 1

    for (ncell, cell) in enumerate(qp[1,:])
        ind = (ncell - 1) * length(cell)
        for (npoint, point) in enumerate(cell)
            val = rfspace(point.point)
            for localbasis in eachindex(val)
                for data in tad.data[:,localbasis,ncell]
                    if data[1] != 0 && ind + npoint != 0
                        rc[sind, 1] = ind + npoint
                        rc[sind, 2] = data[1]
                        vals[sind] = val[localbasis].value * point.weight * data[2]
                        sind += 1
                    end 
                end 
            end
        end
    end

    return rc, vals

end



function getBmatrix(op::BEAST.MaxwellOperator3D, qp::Matrix, X::BEAST.Space)
    rfspace = refspace(X)
    _, tad, _ = assemblydata(X)
    len = length(qp) * length(qp[1,1]) * size(tad.data)[1] * size(tad.data)[2]
    rc = ones(Int, len, 2)
    vals = zeros(Float64, len, 3)
    sind = 1
    for (ncell, cell) in enumerate(qp[1,:])
        ind = (ncell - 1) * length(cell)
        for (npoint, point) in enumerate(cell)
            val = rfspace(point.point)
            for localbasis in eachindex(val)
                for data in tad.data[:,localbasis,ncell]
                    if data[1] != 0 && ind + npoint != 0
                        rc[sind, 1] = ind + npoint
                        rc[sind, 2] = data[1]
                        vals[sind, 1] = val[localbasis].value[1] * point.weight * data[2]
                        vals[sind, 2] = val[localbasis].value[2] * point.weight * data[2]
                        vals[sind, 3] = val[localbasis].value[3] * point.weight * data[2]
                        sind += 1
                    end 
                end 
            end
        end
    end

    return rc, vals

end

function getBmatrix_curl(qp::Matrix, X::BEAST.Space)
    rfspace = refspace(X)
    _, tad, _ = assemblydata(X)
    len = length(qp) * length(qp[1,1]) * size(tad.data)[1] * size(tad.data)[2]
    rc = ones(Int, len, 2)
    vals = zeros(Float64, len, 3)
    sind = 1

    for (ncell, cell) in enumerate(qp[1,:])
        ind = (ncell - 1) * length(cell)
        for (npoint, point) in enumerate(cell)
            val = rfspace(point.point)
            for localbasis in eachindex(val)
                for data in tad.data[:,localbasis,ncell]
                    if data[1] != 0 && ind + npoint != 0
                        rc[sind, 1] = ind + npoint
                        rc[sind, 2] = data[1]
                        vals[sind, 1] = val[localbasis].curl[1] * point.weight * data[2]
                        vals[sind, 2] = val[localbasis].curl[2] * point.weight * data[2]
                        vals[sind, 3] = val[localbasis].curl[3] * point.weight * data[2]
                        sind += 1
                    end 
                end 
            end
        end
    end

    return rc, vals
end

function getBmatrix_div(qp::Matrix, X::BEAST.Space)
    rfspace = refspace(X)
    _, tad, _ = assemblydata(X)
    len = length(qp) * length(qp[1,1]) * size(tad.data)[1] * size(tad.data)[2]
    rc = ones(Int, len, 2)
    vals = zeros(Float64, len)
    sind = 1

    for (ncell, cell) in enumerate(qp[1,:])
        ind = (ncell - 1) * length(cell)
        for (npoint, point) in enumerate(cell)
            val = rfspace(point.point)
            for localbasis in eachindex(val)
                for data in tad.data[:,localbasis,ncell]
                    if data[1] != 0 && ind + npoint != 0
                        rc[sind, 1] = ind + npoint
                        rc[sind, 2] = data[1]
                        vals[sind] = val[localbasis].divergence * point.weight * data[2]
                        sind += 1
                    end 
                end 
            end
        end
    end

    return rc, vals
end

function getnormals(qp::Matrix)
    normals = zeros(Float64, length(qp)*length(qp[1,1]), 3)
    for (i, points) in enumerate(qp[1,:])
        for (j, point) in enumerate(qp[1,i])
            normals[(i-1)*length(points) + j, :] = BEAST.normal(point.point)
        end
    end
    
    return normals
end
 