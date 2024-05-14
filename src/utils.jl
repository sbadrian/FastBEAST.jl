function estimate_norm(mat; tol=1e-4, itmax = 1000)
    v = rand(size(mat,2))

    v = v/norm(v)
    itermin = 3
    i = 1
    σold = 1
    σnew = 1
    @info "Estimate norm"
    while (norm(sqrt(σold)-sqrt(σnew))/norm(sqrt(σold)) > tol || i < itermin) && i < itmax
        @info i, norm(sqrt(σold)-sqrt(σnew))/norm(sqrt(σold))
        σold = σnew
        w = Vector(mat*v)
        x = Vector(adjoint(mat)*w)
        σnew = norm(x)
        v = x/norm(x)
        i += 1
    end
    return sqrt(σnew)
end

function estimate_reldifference(hmat::H, refmat; tol=1e-4) where {F, H <: LinearMaps.LinearMap{F}}
    #if size(hmat) != size(refmat)
    #    error("Dimensions of matrices do not match")
    #end
    
    v = rand(F, size(hmat,2))

    v = v/norm(v)
    itermin = 3
    i = 1
    σold = 1
    σnew = 1
    @info "Estimate norm of reference matrix"
    while norm(sqrt(σold)-sqrt(σnew))/norm(sqrt(σold)) > tol || i < itermin
        @info i, norm(sqrt(σold)-sqrt(σnew))/norm(sqrt(σold))
        σold = σnew
        w = Vector(hmat*v) - Vector(refmat*v)
        x = Vector(adjoint(hmat)*w) - Vector(adjoint(refmat)*w)
        σnew = norm(x)
        v = x/σnew
        i += 1
    end
    @info "Estimate norm of reference matrix"
    norm_refmat = estimate_norm(refmat, tol=tol)

    return sqrt(σnew)/norm_refmat
end