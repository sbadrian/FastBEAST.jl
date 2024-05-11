using ExaFMMt
using BEAST
using LinearAlgebra
using LinearMaps
using SparseArrays
using StaticArrays

#struct FMMOptions{I, K}

struct ExaFMMOptions{I}
    p::I
    ncrit::I
end

function ExaFMMOptions(; p=8, ncrit=50)
    return ExaFMMOptions(p, ncrit)
end

function ExaFMMOptions(tol; ncrit=50)
    # Should compute/choose p accordingly to match tolerance
    error("This function is not yet implemented")
end
"""
    assemble_fmm(
        spoints::Matrix{F},
        tpoints::Matrix{F},
        options::ExaFMMt.FMMOptions;
        computetransposeadjoint=false
    ) where F <: Real

Function calls setup routine of ExaFMM library.

# Arguments
- `spoints::Matrix{F}`: Array of source points in Float64 or Float32.
- `tpoints::Matrix{F}`: Array of target points in Float64 or Float32.
- `options::ExaFMMt.FMMOptions`
"""
function assemble_fmm(
    spoints::Matrix{F},
    tpoints::Matrix{F},
    options::ExaFMMt.FMMOptions;
    computetransposeadjoint=false
) where F <: Real

    @info "ExaFMM: assemble operator"
    A = setup(spoints, tpoints, options)

    if computetransposeadjoint
        @info "ExaFMM: assemble transpose operator"
        Aᵀ = setup(tpoints, spoints, options)
    else
        Aᵀ = A
    end

    return A, Aᵀ
end

"""
    getfullrankblocks(
        operator::BEAST.AbstractOperator,
        test_functions::BEAST.Space,
        trial_functions::BEAST.Space;
        threading=:single,
        nmin=10,
        quadstratcbk=BEAST.DoubleNumQStrat(1,1),
        quadstratfbk=BEAST.defaultquadstrat(operator, test_functions, trial_functions)
    )

Function assembles the fullrankblocks that contain the close interactions of basis and
test functions.
These interactions are inaccurate when computed by the FMM.
The inaccurate values of the FMM are subtracted of the solution using direct evaluations with
the same quadrature strategy as used to sample the input points of the FMM.
The corrected evaluated blocks of the close interactions are added to the solution.

# Arguments
- `operator::BEAST.AbstractOperator`: Operator of BEM.
- `test_functions::BEAST.Space`: Test functions.
- `trial_functions::BEAST.Space`: Trial functions.
- `treeoptions`: Defines which tree is build and the minimal number of basis functions
in each box.
- `multithreading`: Determines weather the computations is single or multithreaded.
- `quadstratcbk`: Quadrature strategy used for the correction of the FMM. 
SafeDoubleNumQStrat() uses a classical Gaussian quadrature and returns zero 
if test equals trial function.
- `quadstratfbk`: Quadrature strategy for correct evaluation of close interactions. 
The `BEAST.defaultquadstrat()` can handle equal test and trial functions.
"""
function getfullrankblocks(
    operator::BEAST.AbstractOperator,
    test_functions::BEAST.Space,
    trial_functions::BEAST.Space;
    treeoptions=BoxTreeOptions(nmin=10),
    multithreading=false,
    quadstratcbk=SafeDoubleNumQStrat(1, 1),
    quadstratfbk=BEAST.defaultquadstrat(operator, test_functions, trial_functions)
)

    @views blkasm = BEAST.blockassembler(
        operator,
        test_functions,
        trial_functions;
        quadstrat=quadstratfbk
    )

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

    test_tree = create_tree(test_functions.pos, treeoptions)
    trial_tree = create_tree(trial_functions.pos, treeoptions)

    fullinteractions = SVector{2}[]
    compressableinteractions = SVector{2}[]

    FastBEAST.computerinteractions!(
        test_tree,
        trial_tree,
        fullinteractions,
        compressableinteractions
    )

    if !multithreading
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
    elseif multithreading
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

"""
    meshtopoints(X::BEAST.Space, quadorder)

Functions samples quadrature points of given `quadorder` on each mesh element.

# Arguments
- `X::BEAST.Space`: Basis function (Contains the mesh).
- `quadorder`: Order of quadrature.
"""
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

"""
    sample_basisfunctions(op::BEAST.Helmholtz3DOp, qp::Matrix, X::BEAST.Space)

Function computes a sparse matrix that convertes the action of test and trial functions 
onto weighted sums over the quadrature points.

# Arguments
- `op::BEAST.Helmholtz3DOp`: BEM operator. 
- `qp::Matrix`: Matrix containing the quadrature points and weights, 
generated by `BEAST.quadpoints()`.
- `X::BEAST.Space`: Basis function.
"""
function sample_basisfunctions(op::BEAST.Helmholtz3DOp, qp::Matrix, X::BEAST.Space)
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


"""
    sample_basisfunctions(op::BEAST.MaxwellOperator3D, qp::Matrix, X::BEAST.Space)

Function computes a sparse matrix that convertes the action of test and trial functions 
onto weighted sums over the quadrature points.

# Arguments
- `op::BEAST.MaxwellOperator3D`: BEM operator. 
- `qp::Matrix`: Matrix containing the quadrature points and weights, generated by `BEAST.quadpoints()`.
- `X::BEAST.Space`: Basis function.
"""
function sample_basisfunctions(op::BEAST.MaxwellOperator3D, qp::Matrix, X::BEAST.Space)
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

"""
    sample_curlbasisfunctions(qp::Matrix, X::BEAST.Space)

Function computes a sparse matrix that convertes the action of test and trial functions 
onto weighted sums over the quadrature points.
Other than `sample_basisfunctions()` this function computes the curl of the funcitons 
for each quadrature point.

# Arguments
- `qp::Matrix`: Matrix containing the quadrature points and weights, 
generated by `BEAST.quadpoints()`.
- `X::BEAST.Space`: Basis function.
""" 
function sample_curlbasisfunctions(qp::Matrix, X::BEAST.Space)
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

"""
    sample_divbasisfunctions(qp::Matrix, X::BEAST.Space)

Function computes a sparse matrix that convertes the action of test and trial functions 
onto weighted sums over the quadrature points.
Other than `sample_basisfunctions()` this function computes divergence of the functions 
    for each quadrature point.

# Arguments
- `qp::Matrix`: Matrix containing the quadrature points and weights, 
generated by `BEAST.quadpoints()`.
- `X::BEAST.Space`: Basis function.
""" 
function sample_divbasisfunctions(qp::Matrix, X::BEAST.Space)
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

"""
    function getnormals(qp::Matrix)

Function computes the normal vector for each quadrature point.

# Arguments
- `qp::Matrix`: Matrix containing the quadrature points and weights, 
generated by `BEAST.quadpoints()`.
"""
function getnormals(qp::Matrix)
    normals = zeros(Float64, length(qp)*length(qp[1,1]), 3)
    for (i, points) in enumerate(qp[1,:])
        for (j, point) in enumerate(qp[1,i])
            normals[(i-1)*length(points) + j, :] = BEAST.normal(point.point)
        end
    end
    
    return normals
end
 