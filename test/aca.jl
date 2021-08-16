##
N = 1000
A = rand(N,N)

fct(x,y) =  A[x,y]

U,S,V = svd(A)

S = [ i < 15 ? 10.0^(-i) : 0.0 for i = 1:N ]

A = U*diagm(S)*V'

U, V = aca_compression(fct, 1:N, 1:N)

@test U*V' ≈ A atol = 1e-14

##
Ns = 200
Nt = 100
A = rand(Nt,Ns)

fct(x,y) =  A[x,y]

U,S,V = svd(A)

S .= 0
S[1:15] = [10.0^(-i) for i = 1:15 ]

A = U*diagm(S)*V'

U, V = aca_compression(fct, 1:Nt, 1:Ns)

@test U*V' ≈ A atol = 1e-14

##
Ns = 100
Nt = 200
A = rand(Nt,Ns)

fct(x,y) =  A[x,y]

U,S,V = svd(A)

S .= 0
S[1:15] = [10.0^(-i) for i = 1:15 ]

A = U*diagm(S)*V'

U, V = aca_compression(fct, 1:Nt, 1:Ns)

@test U*V' ≈ A atol = 1e-14

##
a = [1.0 0.0 0.0 0.0 0.0]
b = [7.0 3.0 1.0 9.0 1.0]
A = a'*b

fct(x,y) =  A[x,y]

U, V = aca_compression(fct, 1:5, 1:5)

@test U*V' == A

##
a = [1.0 -2.0 6.0 4.0 5.0]
b = [7.0 3.0 1.0 9.0 1.0]
A = a'*b

fct(x,y) =  A[x,y]

U, V = aca_compression(fct, 1:5, 1:5)

@test U*V' == A

##
a1 = [1.0 -2.0 6.0 4.0 5.0]
b1 = [7.0 3.0 1.0 9.0 1.0]
a2 = [9.0 3.0 -6.0 2.0 3.0]
b2 = [5.0 4.0 3.0 1.0 -4.0]
#a3 = [10.0 2.0 -6.0 2.0 3.0]
#b3 = [-11.0 4.0 3.0 1.0 -4.0]
A = a1'*b1 + a2'*b2 #+ a3'*b3


fct(x,y) =  A[x,y]
U, V = aca_compression(fct, 1:5, 1:5)

@test U*V' ≈ A atol = 1e-14

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

fct(x,y) =  B[x,y]

U, V = aca_compression(fct, rowindices, colindices)

@test U*V' ≈ A atol = 1e-14

##
A = rand(2,1)

fct(x,y) =  A[x,y]
U, V = aca_compression(fct, 1:2, 1:1)
@test U*V' ≈ A atol = 1e-14

##
A = rand(1,1)

fct(x,y) =  A[x,y]
U, V = aca_compression(fct, 1:1, 1:1)
@test U*V' ≈ A atol = 1e-14