
using BEAST

function hassemble(operator::BEAST.AbstractOperator, test_functions, trial_functions; 
                   compressor=:aca, tol=1e-4, nmin=100, threading=:single)

    @views blkasm = BEAST.blockassembler(operator, test_functions, trial_functions)
    
    @views function assembler(Z, tdata, sdata)
        @views store(v,m,n) = (Z[m,n] += v)
        blkasm(tdata,sdata,store)
    end

    test_tree = create_tree(test_functions.pos, nmin=nmin)
    trial_tree = create_tree(trial_functions.pos, nmin=nmin)

    @time hmat = HMatrix(assembler, test_tree, trial_tree, 
                         compressor=compressor, T=scalartype(operator), tol=tol,
                         threading=threading)
    return hmat
end