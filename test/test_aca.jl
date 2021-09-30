##
N = 1000
A = rand(N,N)

function fct(B, x, y)
    for i in eachindex(x)
        for j in eachindex(y)
            B[i,j] = A[x[i],y[j]]
        end
    end
end

U,S,V = svd(A)

S = [ i < 15 ? 10.0^(-i) : 0.0 for i = 1:N ]

A = U*diagm(S)*V'

U, V = aca_compression(fct, 1:N, 1:N, T=Float64, svdrecompress=false)
@test size(U,2) == 15
@test U*V ≈ A atol = 1e-14

U, V = aca_compression(fct, 1:N, 1:N, T=Float64, svdrecompress=true)
@test size(U,2) == 15
@test U*V ≈ A atol = 1e-14

##
Ns = 200
Nt = 100
A = rand(Nt,Ns)

function fct(B, x, y)
    for i in eachindex(x)
        for j in eachindex(y)
            B[i,j] = A[x[i],y[j]]
        end
    end
end

U,S,V = svd(A)

S .= 0
S[1:15] = [10.0^(-i) for i = 1:15 ]

A = U*diagm(S)*V'

U, V = aca_compression(fct, 1:Nt, 1:Ns, T=Float64, svdrecompress=false)
@test U*V ≈ A atol = 1e-14

U, V = aca_compression(fct, 1:Nt, 1:Ns, T=Float64, svdrecompress=true)
@test U*V ≈ A atol = 1e-14

##
Ns = 100
Nt = 200
A = rand(Nt,Ns)

function fct(B, x, y)
    for i in eachindex(x)
        for j in eachindex(y)
            B[i,j] = A[x[i],y[j]]
        end
    end
end

U,S,V = svd(A)

S .= 0
S[1:15] = [10.0^(-i) for i = 1:15 ]

A = U*diagm(S)*V'

U, V = aca_compression(fct, 1:Nt, 1:Ns, T=Float64)

@test U*V ≈ A atol = 1e-14

##
a = [1.0 0.0 0.0 0.0 0.0]
b = [7.0 3.0 1.0 9.0 1.0]
A = a'*b

function fct(B, x, y)
    for i in eachindex(x)
        for j in eachindex(y)
            B[i,j] = A[x[i],y[j]]
        end
    end
end

U, V = aca_compression(fct, 1:5, 1:5, T=Float64, svdrecompress=false)

@test U*V == A

##
a = [1.0 -2.0 6.0 4.0 5.0]
b = [7.0 3.0 1.0 9.0 1.0]
A = a'*b

function fct(B, x, y)
    for i in eachindex(x)
        for j in eachindex(y)
            B[i,j] = A[x[i],y[j]]
        end
    end
end

U, V = aca_compression(fct, 1:5, 1:5, T=Float64, svdrecompress=false)

@test U*V == A

##
a1 = [1.0 -2.0 6.0 4.0 5.0]
b1 = [7.0 3.0 1.0 9.0 1.0]
a2 = [9.0 3.0 -6.0 2.0 3.0]
b2 = [5.0 4.0 3.0 1.0 -4.0]
#a3 = [10.0 2.0 -6.0 2.0 3.0]
#b3 = [-11.0 4.0 3.0 1.0 -4.0]
A = a1'*b1 + a2'*b2 #+ a3'*b3

function fct(B, x, y)
    for i in eachindex(x)
        for j in eachindex(y)
            B[i,j] = A[x[i],y[j]]
        end
    end
end

U, V = aca_compression(fct, 1:5, 1:5, T=Float64)

@test U*V ≈ A atol = 1e-13

##
Ns = 100
Nt = 200
A = rand(Nt,Ns)

B = zeros(2*Nt, 2*Ns)

U,S,V = svd(A)

S .= 0
S[1:15] = [10.0^(-i) for i = 1:15 ]

A = U*diagm(S)*V'

rowindices = [2*Nt - 2*i + 1 for i=1:Nt]
colindices = [2*Ns - 2*i + 1 for i=1:Ns]

for i = 1:Ns
    for j=1:Nt
        B[end-2*j+1,end-2*i+1] = A[j,i]
    end
end

function fct(C, x, y)
    for i in eachindex(x)
        for j in eachindex(y)
            C[i,j] = B[x[i],y[j]]
        end
    end
end

U, V = aca_compression(fct, rowindices, colindices, T=Float64)

@test U*V ≈ A atol = 1e-14

##
A = rand(2,1)

function fct(B, x, y)
    for i in eachindex(x)
        for j in eachindex(y)
            B[i,j] = A[x[i],y[j]]
        end
    end
end

U, V = aca_compression(fct, 1:2, 1:1, T=Float64)
@test U*V ≈ A atol = 1e-14

##
A = rand(1,1)

function fct(B, x, y)
    for i in eachindex(x)
        for j in eachindex(y)
            B[i,j] = A[x[i],y[j]]
        end
    end
end

U, V = aca_compression(fct, 1:1, 1:1, T=Float64)
@test U*V ≈ A atol = 1e-14

## Double-layer tests
u1 = rand(10,3)
v1 = rand(20,3)
u2 = rand(10,3)
v2 = rand(20,3)
UV1 = u1*v1'
UV2 = u2*v2'*1e-10
A = [zeros(10,10) UV1; UV2' zeros(20,20)]

function fct(B, x, y)
    for i in eachindex(x)
        for j in eachindex(y)
            B[i,j] = A[x[i],y[j]]
        end
    end
end

U, V = aca_compression(fct, 1:30, 1:30, T=Float64, dblsupport=true, svdrecompress=false)
@test U*V ≈ A atol = 1e-14

##
u1 = rand(10,3)
v1 = rand(10,3)
UV1 = u1*v1'
A = [UV1; zeros(20,10)]

function fct(B, x, y)
    for i in eachindex(x)
        for j in eachindex(y)
            B[i,j] = A[x[i],y[j]]
        end
    end
end

U, V = aca_compression(fct, 1:30, 1:10, T=Float64, dblsupport=true, svdrecompress=false)

@test U*V ≈ A atol = 1e-14

##

A = [0.00167465-0.000401981im  -0.00605516+0.00110134im;
-0.00164186+0.000530566im  -0.00582023+0.00170625im;
 0.00283628-0.000816132im  0.000261947-2.70958e-5im;
 0.00121768-0.000243375im    0.0076166-0.00147568im;
 0.00254748-0.000508583im   -0.0114198+0.00135477im;
-0.00244333+0.000318334im   -0.0131399+0.00149213im]

function fct(B, x, y)
    for i in eachindex(x)
        for j in eachindex(y)
            B[i,j] = A[x[i],y[j]]
        end
    end
end

U, V = aca_compression(fct, 1:6, 1:2, tol = 1e-4, T=ComplexF64, dblsupport=false, svdrecompress=false)

@test U*V ≈ A atol = 1e-4

##

A =  K_bc_full[mv.leftindices,mv.rightindices]

function fct(B, x, y)
    for i in eachindex(x)
        for j in eachindex(y)
            B[i,j] = A[x[i],y[j]]
        end
    end
end

U, V = aca_compression(fct, 1:14, 1:6, tol = 1e-4, T=ComplexF64, dblsupport=true, svdrecompress=false)

@test U*V ≈ A atol = 1e-4