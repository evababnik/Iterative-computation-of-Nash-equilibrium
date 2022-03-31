import numpy as np
import math

#1. iterativni algoritem, ki vrne vrednost matrične igre
matrika = np.array([[4, 1, 1, 2], [-9, 6, 5, 4], [-5, 1, 5, 1]])
stevilo_iteracij = 100000
def zaporedje_u(matrika, stevilo_iteracij):
    m = matrika.shape[0]
    n = matrika.shape[1]
    #u =  np.zeros(velikost_matrike)
    #v = np.zeros(velikost_matrike)
    u = [1] * m
    v = [1] * n
    u = np.array(u)
    v = np.array(v)
    k=0
    for k in range(stevilo_iteracij):
        j = np.argmin(u)
        i = np.argmax(v)
        a_i = (matrika[i, 0:m])
        a_j = matrika[0:n, j]
        u = u + a_i
        v = v + a_j
        k = k + 1
    return(u , v)
#print(np.array(zaporedje_u(matrika, stevilo_iteracij)))

def vrednost_igre(matrika, stevilo_iteracij):
    [u, v] = zaporedje_u(matrika, stevilo_iteracij)
    vrednost = np.min(u) / stevilo_iteracij
    vrednost_2 = np.max(v) / stevilo_iteracij
    return(vrednost, vrednost_2)
    

#print(vrednost_igre(matrika, stevilo_iteracij))

#2. iterativni algoritem, ki vrne vrednost matrice igre in Nashevo 
# ravnovesje v obliki vektorja Z = [x_1, ...., x_n, y_1, ..., y_m, v]

#generiramo vektor Z, ki ustreza dolocenim pogojem:

def generiraj_zacetni_vektor(m,n):
    x_0 = np.random.random(m)
    x_0 /= x_0.sum()
    y_0 = np.random.random(n)
    y_0 /= y_0.sum()
    v = np.random.randint(low=0, high=20, size=1, dtype=int)
    x_0 = x_0.tolist()
    y_0 = y_0.tolist()
    v = v.tolist()
    Z = [*x_0, *y_0, *v]
    return(Z)

#print(generiraj_zacetni_vektor(4,5))
def sestavi_vektor_A_i(m,n, i, matrika):
    if i >= 0 and i <= n - 1:
        A_i = []
        for j in range(m):
            a_j_i = matrika[j,i]
            A_i.append(a_j_i)
        for i in range(n):
            A_i.append(0)
        A_i.append(-1)
    else:
        A_i = []
        for j in range(m):
            A_i.append(0)
        for j in range(n):
            a_i_n_j = matrika[i-n, j]
            A_i.append(-a_i_n_j)
        A_i.append(1)
    return(A_i)

def sestavi_vektor_A_0_i(m,n, i):
    A_0_i = []
    if i >= 0 and i <= n-1:
        for i in range(m):
            A_0_i.append(1)
        for i in range(n):
            A_0_i.append(0)
        A_0_i.append(0)
    else:
        for i in range(m):
            A_0_i.append(0)
        for i in range(n):
            A_0_i.append(1)
        A_0_i.append(0)
    return(A_0_i)


#print(sestavi_vektorje_A_i(3, 2, 2, matrika))
def iteracija2(max_stevilo_iteracij, matrika):
    m = matrika.shape[0]
    n = matrika.shape[1]
    Z = np.array(generiraj_zacetni_vektor(m, n))
    korak = 1
    for korak in range(max_stevilo_iteracij):
        produkt_a_z = []
        for i in range(m + n):
            A_i = np.array(sestavi_vektor_A_i(m,n,i,matrika))
            a_i_krat_z = np.dot(A_i, Z)
            produkt_a_z.append(a_i_krat_z)
        if all(elem >= 0 for elem in produkt_a_z):
            break
        else:
            produkt_a_z = np.array(produkt_a_z)
            j_k = np.argmin(produkt_a_z)
            A_0_j_k = np.array(sestavi_vektor_A_0_i(m, n, j_k))
            A_j_k = np.array(sestavi_vektor_A_i(m, n, j_k, matrika))
            b_0_j_k = 1 / (np.dot(A_0_j_k, A_0_j_k) ** (1 / 2))
            B_j_k= A_j_k / (np.dot(A_j_k, A_j_k) ** (1 / 2))
            B_0_j_k = A_0_j_k / (np.dot(A_0_j_k, A_0_j_k) ** (1 / 2))
            cos_theta_jk = (np.dot(A_0_j_k, A_j_k)) / ((np.dot(A_0_j_k, A_0_j_k) ** (1 / 2)) * (np.dot(A_j_k, A_j_k) ** (1 / 2)))
            alfa = - Z * B_j_k * ((1 - cos_theta_jk ** 2) ** (-1))
            beta = b_0_j_k - (Z + alfa * B_j_k) * B_0_j_k
            zacasni_Z  = Z + alfa * B_j_k + beta * B_0_j_k
            b_j_k = 1 / (np.dot(A_j_k, A_j_k) ** (1 / 2))
            zacasni_v = zacasni_Z[-1] - ((-b_j_k) * (np.dot(B_j_k, Z))) / (1 - cos_theta_jk ** 2)
            zacasni_x = []
            vsota_b_j_k = 0
            for h in range(m):
                vsota_b_j_k = vsota_b_j_k + A_j_k[h] / (np.dot(A_j_k, A_j_k) ** (1 / 2))
            for p in range(m):
                b_p_jk = A_j_k[p] / (np.dot(A_j_k, A_j_k) ** (1 / 2))
                x_p = Z[p] - (b_p_jk * (B_j_k * Z)) / (1-cos_theta_jk ** 2) + ((B_j_k * Z) * vsota_b_j_k) / (m * (1 - cos_theta_jk ** 2))
                zacasni_x.append(x_p)
           # print(0 + 1)
           # print(zacasni_v)
           # print(1 +1)
            print(zacasni_x)
            print(zacasni_Z)
           # print( 2 + 1)
           # print(Z )
            preveri_zacasni_Z = Z.tolist()
            if all(elem >= 0 for elem in preveri_zacasni_Z[1:m]):
                Z = zacasni_Z
                korak = korak + 1
            else:
                i = 0
                seznam_indeksov_negativnih_komponent_x = []
                seznam_indeksov_nenegativnih_komponent_x = []
                for komponente_x in preveri_zacasni_Z:
                    i = i + 1
                    if komponente_x < 0:
                        preveri_zacasni_Z[i] = 0
                        seznam_indeksov_negativnih_komponent_x.append(i)
                    else:
                        seznam_indeksov_nenegativnih_komponent_x.append(i)
                vsota_neg_komponent_x_i = 0
                r = len(seznam_indeksov_negativnih_komponent_x)
                for i in seznam_indeksov_negativnih_komponent_x:
                    vsota_neg_komponent_x_i = vsota_neg_komponent_x_i + preveri_zacasni_Z[i]
                for i in seznam_indeksov_nenegativnih_komponent_x:
                    x_i = preveri_zacasni_Z[i]
                    vsota = x_i + vsota_neg_komponent_x_i / (m - r)
                    if vsota < 0:
                        preveri_zacasni_Z[i] = 0
                Z = np.array(preveri_zacasni_Z)
                korak = korak + 1
    
    return(Z)           


m = 3
n = 3
#a = np.array(sestavi_vektorje_A_i(m, n, 2, matrika))
#b = np.array(generiraj_zacetni_vektor(m,n)) 
#print(np.dot(a,b))
iteracija2(3, matrika)
#print(vrednost_igre(matrika, 10000 ))
