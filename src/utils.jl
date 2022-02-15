function estimate_norm(mat; tol=1e-4, itmax = 1000)
    v = rand(size(mat,2))

    v = v/norm(v)
    itermin = 3
    i = 1
    σold = 1
    σnew = 1
    while (norm(sqrt(σold)-sqrt(σnew))/norm(sqrt(σold)) > tol || i < itermin) && i < itmax
        σold = σnew
        w = mat*v
        x = adjoint(mat)*w
        σnew = norm(x)
        v = x/norm(x)
        i += 1
    end
    return sqrt(σnew)
end

function estimate_reldifference(hmat::H, refmat::M; tol=1e-4) where {F, H <: LinearMaps.LinearMap{F}, M <: AbstractMatrix{F}}
    #if size(hmat) != size(refmat)
    #    error("Dimensions of matrices do not match")
    #end
    
    v = rand(F, size(hmat,2))

    v = v/norm(v)
    itermin = 3
    i = 1
    σold = 1
    σnew = 1
    while norm(sqrt(σold)-sqrt(σnew))/norm(sqrt(σold)) > tol || i < itermin
        σold = σnew
        w = hmat*v - refmat*v
        x = adjoint(hmat)*w - adjoint(refmat)*w
        σnew = norm(Vector(x))
        v = x/σnew
        i += 1
    end

    norm_refmat = estimate_norm(refmat, tol=tol)

    return sqrt(σnew)/norm_refmat
end