using BEAST

function hassemble(
    operator::BEAST.AbstractOperator,
    test_functions,
    trial_functions;
    compressor=:aca,
    tol=1e-4,
    treeoptions=BoxTreeOptions(nmin=100),
    maxrank=200,
    threading=:single,
    quadstrat=BEAST.defaultquadstrat(operator, test_functions, trial_functions),
    verbose=false,
    svdrecompress=false
)

    @views blkasm = BEAST.blockassembler(operator, test_functions, trial_functions)
    
    @views function assembler(Z, tdata, sdata)
        @views store(v,m,n) = (Z[m,n] += v)
        blkasm(tdata,sdata,store)
    end

    @views farblkasm = BEAST.blockassembler(
        operator,
        test_functions,
        trial_functions,
        quadstrat=quadstrat
    )
    
    @views function farassembler(Z, tdata, sdata)
        @views store(v,m,n) = (Z[m,n] += v)
        farblkasm(tdata,sdata,store)
    end


    test_tree = create_tree(test_functions.pos, treeoptions)
    trial_tree = create_tree(trial_functions.pos, treeoptions)

    @time hmat = HMatrix(
        assembler,
        test_tree,
        trial_tree,
        Int64,
        scalartype(operator),
        compressor=compressor,
        tol=tol,
        maxrank=maxrank,
        threading=threading,
        farmatrixassembler=farassembler,
        verbose=verbose,
        svdrecompress=svdrecompress
    )

    return hmat
end

function fmmassemble(
    operator::BEAST.AbstractOperator,
    test_functions::BEAST.LagrangeBasis,
    trial_functions::BEAST.LagrangeBasis;
    nmin=10,
    threading=:single,
    npoints=3,
    fmmoptions=LaplaceFMMOptions()
)
    fullrankblocks, correctionblocks, _ = getfullrankblocks(
        operator,
        test_functions,
        trial_functions,
        nmin=nmin,
        threading=threading,
        quadstratcbk=SafeDoubleNumQStrat(npoints, npoints)
    )
    
    fullmat  = fullmatrix(fullrankblocks)
    BtCB = fullmatrix(correctionblocks)
    
    points, qp = meshtopoints(test_functions, npoints)
    B = getBmatrix(qp, test_functions)

    fmm = assemble_fmm(
        points,
        points,
        fmmoptions
    )

    if operator isa BEAST.HH3DDoubleLayer

        normals = zeros(Float64, length(qp)*length(qp[1,1]), 3)
        for (i, points) in enumerate(qp[1,:])
            for (j, point) in enumerate(qp[1,i])
                normals[(i-1)*length(points) + j, :] = normal(point.point)
            end
        end

        return FMMMatrixDL(
            fmm,
            normals,
            B,
            sparse(transpose(B)),
            BtCB,
            fullmat,
            size(fullmat)[1],
            size(fullmat)[2]
        )
    
    elseif operator isa BEAST.HH3DDoubleLayerTransposed 

        normals = zeros(Float64, length(qp)*length(qp[1,1]), 3)
        for (i, points) in enumerate(qp[1,:])
            for (j, point) in enumerate(qp[1,i])
                normals[(i-1)*length(points) + j, :] = normal(point.point)
            end
        end

        return FMMMatrixADL(
            fmm,
            normals,
            B,
            sparse(transpose(B)),
            BtCB,
            fullmat,
            size(fullmat)[1],
            size(fullmat)[2]
        )

    elseif operator isa BEAST.HH3DHyperSingularFDBIO
        
        normals = zeros(Float64, length(qp)*length(qp[1,1]), 3)
        for (i, points) in enumerate(qp[1,:])
            for (j, point) in enumerate(qp[1,i])
                normals[(i-1)*length(points) + j, :] = normal(point.point)
            end
        end

        Bcurl1 = getBmatrix_curl(qp, test_functions, 1)
        Bcurl2 = getBmatrix_curl(qp, test_functions, 2)
        Bcurl3 = getBmatrix_curl(qp, test_functions, 3)
        
        return FMMMatrixHS(
            fmm,
            normals,
            Bcurl1,
            Bcurl2,
            Bcurl3,
            sparse(transpose(Bcurl1)),
            sparse(transpose(Bcurl2)),
            sparse(transpose(Bcurl3)),
            B,
            sparse(transpose(B)),
            BtCB,
            fullmat,
            size(fullmat)[1],
            size(fullmat)[2]
        )

    else
        
        return FMMMatrixSL(
            fmm,
            B,
            sparse(transpose(B)),
            BtCB,
            fullmat,
            size(fullmat)[1],
            size(fullmat)[2]
        )
    end
end

# The following to function ensure that no dynamic dispatching is
# performed since we know already that all triangles are well-separate

# Copied from BEAST/examples/quadstrat.jl
function BEAST.quaddata(op, tref, bref,
    tels, bels, qs::BEAST.DoubleNumQStrat)

    qs = BEAST.DoubleNumWiltonSauterQStrat(qs.outer_rule, qs.inner_rule, 1, 1, 1, 1, 1, 1)
    BEAST.quaddata(op, tref, bref, tels, bels, qs)
end

# Copied from BEAST/examples/quadstrat.jl
function BEAST.quadrule(op, tref, bref,
    i ,τ, j, σ, qd, qs::BEAST.DoubleNumQStrat)

    return BEAST.DoubleQuadRule(
        qd.test_qp[1,i],
        qd.bsis_qp[1,j])
end


# Safe evaluation of Greens function
struct SafeDoubleNumQStrat{R}
    outer_rule::R
    inner_rule::R
end

struct SafeDoubleQuadRule{P,Q}
    outer_quad_points::P
    inner_quad_points::Q
end

function BEAST.quadrule(op, tref, bref, i ,τ, j, σ, qd, qs::SafeDoubleNumQStrat)
    return SafeDoubleQuadRule(
        qd.test_qp[1,i],
        qd.bsis_qp[1,j])
end

function BEAST.quaddata(
    op::BEAST.Helmholtz3DOp,
    test_refspace::BEAST.LagrangeRefSpace,
    trial_refspace::BEAST.LagrangeRefSpace,
    test_elements,
    trial_elements,
    qs::SafeDoubleNumQStrat
)

    test_eval(x)  = test_refspace(x,  Val{:withcurl})
    trial_eval(x) = trial_refspace(x, Val{:withcurl})

    test_qp = BEAST.quadpoints(test_eval,  test_elements,  (qs.outer_rule,))
    bsis_qp = BEAST.quadpoints(trial_eval, trial_elements, (qs.inner_rule,))

    return (;test_qp, bsis_qp)
end

function BEAST.momintegrals!(biop, tshs, bshs, tcell, bcell, z, strat::SafeDoubleQuadRule)
    
    igd = BEAST.Integrand(biop, tshs, bshs, tcell, bcell)
    womps = strat.outer_quad_points
    wimps = strat.inner_quad_points
    
    for womp in womps
        tgeo = womp.point
        tvals = womp.value
        M = length(tvals)
        jx = womp.weight
        
        for wimp in wimps
            bgeo = wimp.point
            bvals = wimp.value
            N = length(bvals)
            jy = wimp.weight

            j = jx * jy

            if !(bgeo.cart ≈ tgeo.cart)
                z1 = j * igd(tgeo, bgeo, tvals, bvals)
                for n in 1:N
                    for m in 1:M
                        z[m,n] += z1[m,n]
            end end end
        end
    end

    return z
end