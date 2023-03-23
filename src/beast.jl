using BEAST

function hassemble(
    operator::BEAST.AbstractOperator,
    test_functions,
    trial_functions;
    compressor=:aca,
    acaoptions=ACAOptions(),
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
        acaoptions=acaoptions,
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
    test_functions::BEAST.Space,
    trial_functions::BEAST.Space;
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
    K = scalartype(operator)
    fullmat = HMatrix(
        fullrankblocks,
        MatrixBlock{Int, K, LowRankMatrix{K}}[],
        length(test_functions.fns),
        length(trial_functions.fns),
        0,
        0,
        threading == :multi ?  true : false
    )
    BtCB = HMatrix(
        correctionblocks,
        MatrixBlock{Int, K, LowRankMatrix{K}}[],
        length(test_functions.fns),
        length(trial_functions.fns),
        0,
        0,
        threading == :multi ?  true : false
    )
    
    testpoints, testqp = meshtopoints(test_functions, npoints)
    trialpoints, trialqp = meshtopoints(trial_functions, npoints)

    fmm = assemble_fmm(
        trialpoints,
        testpoints,
        options=fmmoptions
    )

    return FMMMatrix(
        operator,
        test_functions, 
        trial_functions, 
        testqp,
        trialqp,
        fmm,
        BtCB,
        fullmat,
    )
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

#function BEAST.quadrule(op, tref, bref,
#    i ,τ, j, σ, qd, qs::BEAST.DoubleNumQStrat)
#
#    return BEAST.DoubleQuadRule(
#        qd.tpoints[1,i],
#        qd.bpoints[1,j])
#end


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

function BEAST.quaddata(
    op::BEAST.MaxwellOperator3D,
    test_refspace::BEAST.RTRefSpace,
    trial_refspace::BEAST.RTRefSpace,
    test_elements,
    trial_elements,
    qs::SafeDoubleNumQStrat
)

    test_eval(x)  = test_refspace(x)
    trial_eval(x) = trial_refspace(x)

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