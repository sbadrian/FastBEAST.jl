using FastBEAST

function test_fullrank(N)
    store = Vector{MatrixView}(undef, N)
    n = 100
    for i=1:N
        store[i] = FullMatrixView(rand(n,n), [1; 2], [1; 2], 10, 20)
    end

    return store
end

function test_fullrank2(N)
    store = MatrixView[]
    n = 100
    for i=1:N
        push!(store, FullMatrixView(rand(n,n), [1; 2], [1; 2], 10, 20))
    end

    return store
end

function test_fullrank3(N)
    store = Vector{FullMatrixView}(undef, N)
    n = 100
    for i=1:N
        store[i] = FullMatrixView(rand(n,n), [1; 2], [1; 2], 10, 20)
    end

    return store
end

function test_lowrank(N)
    store = Vector{MatrixView}(undef, N)
    n = 100
    for i=1:N
        store[i] = LowRankMatrixView(rand(n,n), rand(n,n), [1; 2], [1; 2], 10, 20)
    end

    return store
end

function test_lowrank2(N)
    store = MatrixView[]
    n = 100
    for i=1:N
        push!(store, LowRankMatrixView(rand(n,n), rand(n,n), [1; 2], [1; 2], 10, 20))
    end

    return store
end

function test_lowrank3(N)
    store = Vector{LowRankMatrixView}(undef, N)
    n = 100
    for i=1:N
        store[i] = LowRankMatrixView(rand(n,n), rand(n,n), [1; 2], [1; 2], 10, 20)
    end

    return store
end

function test_all(N)
@time    A = test_fullrank(N)
@time    A = test_fullrank2(N)
@time    A = test_fullrank3(N)
@time    A = test_lowrank(N)
@time    A = test_lowrank2(N)
@time    A = test_lowrank3(N)
end