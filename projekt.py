import numpy as np

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

def vrednost_igre(matrika, stevilo_iteracij):
    [u, v] = zaporedje_u(matrika, stevilo_iteracij)
    vrednost = np.min(u) / stevilo_iteracij
    vrednost_2 = np.max(v) / stevilo_iteracij
    return(vrednost, vrednost_2)
    

print(vrednost_igre(matrika, stevilo_iteracij))