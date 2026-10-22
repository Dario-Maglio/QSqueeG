##########################################

### Construction of initial states

############################################

function binom(k, n)
    """Compute binomial coefficients"""
    j = min(k, n-k)
    res = 0
    for i in 1:j
        res += log(n-i+1)-log(i)
    end
    res = exp(res/2)
    res
end;


function binom3(k, n)
    #contains all the terms for the CSS x state
    #attention racine carrée déjà incluse, et un terme en 1/sqrt(2)^{2S} !
    j = min(k, n-k)
    res = 0
    for i in 1:j
        res += log(n-i+1)-log(i)-log(2)
    end
    for i in 1:n-j
        res -= log(2)
    end
    res = exp(res/2)
end;

function init_cssX(S)
    #coherent spin state along +x in the z basis
    #binom3 renvoie déjà la racine carrée !
    # de +S à -S
    n = round(Int, 2*S+1)
    res = zeros(Float64, n)
    for k in 0:n-1
        res[n-k] = binom3(k, n-1)
    end
    res
end;

function init_cssXm(S)
    #coherent spin state along -x in the z basis
    #binom3 renvoie déjà la racine carrée !
    n = round(Int, 2*S+1)
    res = zeros(Float64, n)
    for k in 0:n-1
        res[n-k] = ((-1)^(n-1-k))*binom3(k, n-1)
    end
    res
end;


function init_css(S, theta, phi)
    #general coherent spin state in the z basis
    n = round(Int, 2*S+1)
    res = zeros(ComplexF64, n)
    for k in 0:n-1
        #k = S-m
        kp = 2*S-k #kp = S+m
        res[n-k] = binom(k, n-1)*sin(theta/2)^(kp)*cos(theta/2)^(k)*exp(-(1im*phi*k))
    end
    res
end;


function init_Dickez(S, k)
    #Dicke state
    #k is ranging from 1 to 2S+1, corresponding to S to -S for the projection along z
    n = round(Int, 2*S+1)
    res = zeros(Float64, n)
    res[k] = 1
    res
end

function baseX(S)
    #matrice de passage pour écrire un spin dans la base propre de Sx
    P = eigvecs(Sx(S))
    # vp = eigvals(Sx(S))
    #println(vp)
    P
end;

function baseXN(N, S)
    #matrice de passage pour écrire psi dans la base des états de Dicke selon X
    #attention en l'état il faut multiplier psi par base' pour obtenir psiX
    BX = baseX(S)
    res = BX
    for i in 1:N-1
        res = prod_tens_op(res, BX)      
    end
    res
end;

function init_Dickex(S, k, base)
    #k is ranging from 1 to 2S+1, corresponding to S to -S for the projection along x
    #warning : baseX is constructed the other way around (due to diagonalization) -> take n+1-k
    n = round(Int, 2*S+1)
    res = zeros(Float64, n)
    res[n+1-k] = 1
    res = base*res #move to x eigenstate
    res
end  

function norme(psi)
    #normalize a vector
    val = sqrt(real(psi' * psi))
    psi/val
end;

##########################################

### Construction of local spin operators

############################################


function Cp(S, m)
    res = sqrt(S*(S+1) - m*(m+1))
    res
end;

function Cm(S, m)
    sqrt(S*(S+1) - m*(m-1))
end;


function identite(n)
    Id = zeros(Float64, n, n)
    for i in 1:n
        Id[(i-1)*n + i] = 1
    end
    Id
end;

function Sz(S)
    #from S to -S
    n = round(Int, 2*S+1)
    res = zeros(Float64, n, n)
    for i in 1:n
        m = S+1-i
        res[i, i] = m
    end
    res
end;

function Sx(S)
    #from S to -S
    n = round(Int, 2*S+1)
    res = zeros(Float64, n, n)
    for i in 1:n-1
        m = S+1-i
        res[i+1, i] = Cp(S, m-1)/2
        res[i, i+1] = Cp(S, m-1)/2
    end
    res
end;


function Splus(S)
    #from S to -S
    n = round(Int, 2*S+1)
    res = zeros(Float64, n, n)
    for i in 1:n-1
        m = S+1-i
        res[i*n + i] = Cp(S, m-1)
    end
    res
end;

function Sminus(S)
    #from S to -S
    n = round(Int, 2*S+1)
    res = zeros(Float64, n, n)
    for i in 1:n-1
        m = S+1-i
        res[(i-1)*n + i+1] = Cp(S, m-1)
    end
    res
end;

function Sy(S)
    #from S to -S
    #this one is a complex matrix
    n = round(Int, 2*S+1)
    res = zeros(ComplexF64, n, n)
    for i in 1:n-1
        m = S+1-i
        res[i*n + i] = -Cp(S, m-1)*complex(0, 1)/2
        res[(i-1)*n + i+1] = Cp(S, m-1)*complex(0, 1)/2
    end
    res
end;


##########################################

### Construction of Hamiltonien matrix

############################################

function prod_tens_vec(V1, V2)
    #tensorial product of two real vectors
    n = size(V1)[1]
    m = size(V2)[1]
    res = zeros(Float64, n*m)
    for i in 0:n*m-1
        iq = div(i, m)
        ir = mod(i, m)
        res[i + 1] = V1[iq + 1] * V2[ir + 1]
    end
    res
end;

function prod_tens_vecC(V1, V2)
    #tensorial product of two complex vectors
    n = size(V1)[1]
    m = size(V2)[1]
    res = zeros(ComplexF64, n*m)
    for i in 0:n*m-1
        iq = div(i, m)
        ir = mod(i, m)
        res[i + 1] = V1[iq + 1] * V2[ir + 1]
    end
    res
end;

function prod_tens_op(Op1, Op2)
    #tensorial product of two real matrices
    n = size(Op1)[1]
    m = size(Op2)[1]
    res = zeros(Float64, n*m, n*m)
    for i in 0:n*m-1
        iq = div(i, m)
        ir = mod(i, m)
        for j in 0:n*m-1
            jq = div(j, m)
            jr = mod(j, m)
            #println([m, n, iq, jq, ir, jr, iq + jq*m + 1, ir + jr*n + 1])
            res[j*n*m + i + 1] = Op1[iq + jq*n + 1] * Op2[ir + jr*m + 1]
        end
    end
    res
end;

function prod_tens_opC(Op1, Op2)
    #tensorial product of two complex matrices
    n = size(Op1)[1]
    m = size(Op2)[1]
    res = zeros(ComplexF64, n*m, n*m)
    for i in 0:n*m-1
        iq = div(i, m)
        ir = mod(i, m)
        for j in 0:n*m-1
            jq = div(j, m)
            jr = mod(j, m)
            #println([m, n, iq, jq, ir, jr, iq + jq*m + 1, ir + jr*n + 1])
            res[j*n*m + i + 1] = Op1[iq + jq*n + 1] * Op2[ir + jr*m + 1]
        end
    end
    res
end;


function ham_part_lin_diff(N, Op1, i, lesS)
    #construct the tensorial product of Op1 corresponding to subsystem i, with N subsystems of respective size lesS[i]
    #convention : i > j
    #for various spin length on the different sites
    #smaller index on the left 
    ni = 1
    for k in 1:i  
        ni *= round(Int, 2*lesS[k]+1)
    end
    nNi = 1
    for k in i+2:N
        nNi *= round(Int, 2*lesS[k]+1)
    end
    res = prod_tens_op(prod_tens_op(identite(ni), Op1), identite(nNi))
    res
end;

function ham_part_quad_diff(N, Op1, Op2, i, j, lesS)
    #construct the tensorial product of Op1 and Op2 corresponding to subsystem i and j, with N subsystems of respective size lesS[i]
    #convention : i > j
    #for various spin length on the different sites
    #Op 2 avant Op 1
    nj = 1
    for k in 1:j  
        nj *= round(Int, 2*lesS[k]+1)
    end
    nij = 1
    for k in j+2:i
        nij *= round(Int, 2*lesS[k]+1)
    end
    nNi = 1
    for k in i+2:N
        nNi *= round(Int, 2*lesS[k]+1)
    end

    Idj = identite(nj)
    Idij = identite(nij)
    IdNi = identite(nNi)
    res = prod_tens_op(prod_tens_op(prod_tens_op(prod_tens_op(Idj, Op2), Idij), Op1), IdNi)
    res
end;

##########################################

### Linear Algebra and evolution

############################################



function diagonalisation(H)
    #diagnonalization of the Hamiltonian matrix
    eigv = real(eigvals(H))
    P = eigvecs(H)
    Pm = inv(P) #matrice inverse des vecteurs propres
    eigv, P, Pm
    
end;

function evolution(eigvals, P, Pm, psi, t)
    #evolution at time t of the initial state psi, given the eigenvalues and eigenvectors of H
    coef =  Pm * psi #décomposition de psi dans la base des vecteurs propres
    n = length(coef)
    evol = zeros(Complex{Float64}, n) #evolution des coef dans la base propre
    for k in 1:n
        evol[k] = coef[k]*exp(-1im*t*eigvals[k])
    end
    final = P*evol #retour à la base canonique après évolution
    final
end;

function evol_diag(eigvals, psi, t)
    #evolution at time t of initial state psi with a diagonal Hamiltonian whose eigenvalues are given by eigvals
    n = length(psi)
    evol = zeros(ComplexF64, n) #evolution des coef dans la base propre
    for k in 1:n
        evol[k] = psi[k]*exp(-1im*t*eigvals[k])
    end
    evol
end

