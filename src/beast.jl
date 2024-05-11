using BEAST

function hassemble(
    operator::BEAST.AbstractOperator,
    test_functions,
    trial_functions;
    treeoptions=BoxTreeOptions(nmin=100),
    compressor=ACAOptions(),
    quadstratcbk=BEAST.defaultquadstrat(operator, test_functions, trial_functions),
    quadstratfbk=BEAST.defaultquadstrat(operator, test_functions, trial_functions),
    multithreading=false,
    verbose=false
)

    @views blkasm = BEAST.blockassembler(
        operator,
        test_functions,
        trial_functions,
        quadstrat=quadstratfbk
    )
    
    @views function assembler(Z, tdata, sdata)
        @views store(v,m,n) = (Z[m,n] += v)
        blkasm(tdata,sdata,store)
    end

    @views farblkasm = BEAST.blockassembler(
        operator,
        test_functions,
        trial_functions,
        quadstrat=quadstratcbk
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
        farmatrixassembler=farassembler,
        compressor=compressor,
        multithreading=multithreading,
        verbose=verbose
    )
    return hmat
end

function fmmassemble(
    operator,
    fmmoptions,
    test_functions::BEAST.Space,
    trial_functions::BEAST.Space;
    treeoptions=BoxTreeOptions(nmin=10),
    quadstratcbk=SafeDoubleNumQStrat(3, 3),
    quadstratfbk=BEAST.defaultquadstrat(operator, test_functions, trial_functions),
    multithreading=false,
    computetransposeadjoint=false,
    verbose=false
)
    fullrankblocks, correctionblocks, _ = getfullrankblocks(
        operator,
        test_functions,
        trial_functions,
        treeoptions=treeoptions,
        multithreading=multithreading,
        quadstratcbk=quadstratcbk,
        quadstratfbk=quadstratfbk
    )
    K = scalartype(operator)
    fullmat = HMatrix(
        fullrankblocks,
        MatrixBlock{Int, K, LowRankMatrix{K}}[],
        length(test_functions.fns),
        length(trial_functions.fns),
        0,
        0,
        multithreading
    )
    BtCB = HMatrix(
        correctionblocks,
        MatrixBlock{Int, K, LowRankMatrix{K}}[],
        length(test_functions.fns),
        length(trial_functions.fns),
        0,
        0,
        multithreading
    )
    
    testpoints, testqp = meshtopoints(test_functions, quadstratcbk.outer_rule)
    trialpoints, trialqp = meshtopoints(trial_functions, quadstratcbk.outer_rule)

    fmm, fmm_t = assemble_fmm(
        trialpoints,
        testpoints,
        fmmoptions,
        computetransposeadjoint=computetransposeadjoint
    )

    return FMMMatrix(
        operator,
        test_functions, 
        trial_functions, 
        testqp,
        trialqp,
        fmm,
        fmm_t,
        BtCB,
        fullmat,
    )
end

exafmmoptions(gamma::T, fmm) where {T <: Val{0}} = LaplaceFMMOptions(; p=fmm.p, ncrit=fmm.ncrit)
#TODO: Write unit tests for the ModifiedHelmholtzFMMOptions
exafmmoptions(gamma::T, fmm) where {T <: Real} = ModifiedHelmholtzFMMOptions(gamma; p=fmm.p, ncrit=fmm.ncrit)
exafmmoptions(gamma::T, fmm) where {T <: Complex} = HelmholtzFMMOptions(-gamma/im; p=fmm.p, ncrit=fmm.ncrit)


function fmmassemble(
    operator::BEAST.MaxwellOperator3D{T,K},
    test_functions::BEAST.Space,
    trial_functions::BEAST.Space;
    treeoptions=BoxTreeOptions(nmin=10),
    fmmoptions=ExaFMMOptions(),
    quadstratcbk=SafeDoubleNumQStrat(3, 3),
    quadstratfbk=BEAST.defaultquadstrat(operator, test_functions, trial_functions),
    multithreading=false,
    computetransposeadjoint=false,
    verbose=false
) where {T, K}

    return fmmassemble(
        operator,
        exafmmoptions(operator.gamma, fmmoptions),
        test_functions,
        trial_functions;
        treeoptions=treeoptions,
        quadstratcbk=quadstratcbk,
        quadstratfbk=quadstratfbk,
        multithreading=multithreading,
        computetransposeadjoint=computetransposeadjoint,
        verbose=verbose
    )
end

function fmmassemble(
    operator::BEAST.Identity,
    test_functions::BEAST.Space,
    trial_functions::BEAST.Space;
    treeoptions=BoxTreeOptions(nmin=10),
    fmmoptions=ExaFMMOptions(),
    quadstratcbk=SafeDoubleNumQStrat(3, 3),
    quadstratfbk=BEAST.defaultquadstrat(operator, test_functions, trial_functions),
    multithreading=false,
    computetransposeadjoint=false,
    verbose=false
)
    return assemble(operator, test_functions, trial_functions)
end

# The following to function ensure that no dynamic dispatching is
# performed since we know already that all triangles are well-separate

# Copied from BEAST/examples/quadstrat.jl
function BEAST.quaddata(op, tref, bref,
    tels, bels, qs::BEAST.DoubleNumQStrat)

    qs = BEAST.DoubleNumWiltonSauterQStrat(qs.outer_rule, qs.inner_rule, 1, 1, 1, 1, 1, 1)
    BEAST.quaddata(op, tref, bref, tels, bels, qs)
end

function BEAST.quadrule(op, tref, bref,
    i ,τ, j, σ, qd, qs::BEAST.DoubleNumQStrat)

    return BEAST.DoubleQuadRule(
        qd.tpoints[1,i],
        qd.bpoints[1,j])
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
        qd.tpoints[1,i],
        qd.bpoints[1,j])
end

function BEAST.quaddata(
    op::BEAST.Helmholtz3DOp,
    test_refspace::BEAST.LagrangeRefSpace,
    trial_refspace::BEAST.LagrangeRefSpace,
    test_elements,
    trial_elements,
    qs::SafeDoubleNumQStrat
)

    test_eval(x)  = test_refspace(x)
    trial_eval(x) = trial_refspace(x)

    tpoints = BEAST.quadpoints(test_eval,  test_elements,  (qs.outer_rule,))
    bpoints = BEAST.quadpoints(trial_eval, trial_elements, (qs.inner_rule,))

    return (;tpoints, bpoints)
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

    tpoints = BEAST.quadpoints(test_eval,  test_elements,  (qs.outer_rule,))
    bpoints = BEAST.quadpoints(trial_eval, trial_elements, (qs.inner_rule,))

    return (;tpoints, bpoints)
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

function fmmassemble(op::BEAST.LinearCombinationOfOperators, X::BEAST.Space, Y::BEAST.Space;
    fmmoptions=ExaFMMOptions(),
    treeoptions=BoxTreeOptions(nmin=10),
    quadstratcbk=SafeDoubleNumQStrat(3, 3),
    quadstratfbk=BEAST.defaultquadstrat(op, X, Y),
    multithreading=false,
    verbose=false
)

    T = scalartype(op, X, Y)

    M = numfunctions(X)
    N = numfunctions(Y)
    A = BEAST.ZeroMap{T}(1:M, 1:N)

    counter = 0
    for (α,term) in zip(op.coeffs, op.ops)
        counter += 1
        println("Compress operator: ", counter)
        A = A + α * fmmassemble(term, X, Y;
            treeoptions=treeoptions,
            fmmoptions=fmmoptions,
            quadstratcbk=quadstratcbk,
            quadstratfbk=quadstratfbk,
            multithreading=multithreading,
            verbose=verbose
        )
    end

    return A
end