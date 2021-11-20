
using BEAST

function hassemble(
    operator::BEAST.AbstractOperator,
    test_functions,
    trial_functions;
    compressor=:aca,
    treeoptions=BoxTreeOptions(nmin=100),
    tol=1e-4,
    maxrank=100,
    threading=:single,
    farquaddata=BEAST.quaddata,
    verbose=false,
    svdrecompress=true)

    @views blkasm = BEAST.blockassembler(operator, test_functions, trial_functions)
    
    @views function assembler(Z, tdata, sdata)
        @views store(v,m,n) = (Z[m,n] += v)
        blkasm(tdata,sdata,store)
    end

    @views farblkasm = BEAST.blockassembler(
        operator,
        test_functions,
        trial_functions,
        quaddata=farquaddata
    )
    
    @views function farassembler(Z, tdata, sdata)
        @views store(v,m,n) = (Z[m,n] += v)
        farblkasm(tdata,sdata,store)
    end


    test_tree = create_tree(test_functions.pos, treeoptions=treeoptions)
    trial_tree = create_tree(trial_functions.pos, treeoptions=treeoptions)

    @time hmat = HMatrix(assembler, test_tree, trial_tree, 
                         compressor=compressor, T=scalartype(operator), tol=tol, maxrank=maxrank,
                         threading=threading, farmatrixassembler=farassembler, verbose=verbose,
                         svdrecompress=svdrecompress)
    return hmat
end