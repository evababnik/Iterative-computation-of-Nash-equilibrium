import numpy as np
#1. iterativni algoritem, ki vrne vrednost matrične igre
matrika = np.array([[4, 1, 2], [-9, 6, -9], [-5, 1, -34]])
stevilo_iteracij = 100000
def zaporedje_u(matrika, stevilo_iteracij):
    velikost_matrike = matrika.shape[0]
    #u =  np.zeros(velikost_matrike)
    #v = np.zeros(velikost_matrike)
    u = np.array([1, 6,2])
    v = np.array([2, 3, 5])
    k=0
    for k in range(stevilo_iteracij):
        j = np.argmin(u)
        i = np.argmax(v)
        a_i = (matrika[i, 0:velikost_matrike])
        a_j = matrika[0:velikost_matrike, j]
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

print(generiraj_zacetni_vektor(4,5))
def sestavi_vektorje_A_i(m,n, i, matrika):
    if i >= 1 and i <= n:
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

#print(sestavi_vektorje_A_i(3, 2, 2, matrika))
def iteracija2(m,n, max_stevilo_iteracij, matrika):
    Z = np.array(generiraj_zacetni_vektor(m, n))
    produkt_a_z = []
    for i in range(m+n):
        A_i = np.array(sestavi_vektorje_A_i(m,n,i,matrika))
        a_i_krat_z = np.dot(A_i, Z)
        produkt_a_z.append(a_i_krat_z)
    if all(elem >= 0 for elem in produkt_a_z):
        return(Z)
    else:


#m = 3
#n = 2
#a = np.array(sestavi_vektorje_A_i(m, n, 2, matrika))
#b = np.array(generiraj_zacetni_vektor(m,n)) 
#print(np.dot(a,b))
print(iteracija2(3, 2, 2, matrika))

    