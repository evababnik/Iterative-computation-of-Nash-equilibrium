from cmath import sqrt
from statistics import mean
import numpy as np
import math
import cmath
import scipy
from scipy.optimize import minimize
from sympy import Nor
import time

matrika11 = np.array([[4, 3], [2, 4], [5, 2]])
matrika = np.array([[0, -1, -2, 0, 0, 2, 1], [1, 0, 0, 4, 0, 0, -1], [2, 0, 0 , -1, 1, 0, -1], [0, -4, 1, 0, 0, -1, 1], [0, 0, -1, 0 , 0, -3, 1], [-2, 0, 0, 1, 3, 0, -1], [-1, 1, 1, -1, - 1, 1, 0]])

#1. iterativni algoritem, ki vrne vrednost matrične igre

def zaporedje_u(matrika, stevilo_iteracij):
    m = matrika.shape[0]
    n = matrika.shape[1]
    u = [abs(np.random.randint(low=np.min(matrika), high=np.max(matrika), size=1, dtype=int))]* n
    v = [abs(np.random.randint(low=np.min(matrika), high=np.max(matrika), size=1, dtype=int))] * m
    u = np.array(u)
    v = np.array(v)
    k=0
    for k in range(stevilo_iteracij):
        j = np.argmin(u)
        i = np.argmax(v)
        a_i = (matrika[i, 0:n])
        a_j = matrika[0:m, j]
        u = u + a_i
        v = v + a_j
        k = k + 1
    return(u , v)


def vrednost_igre(matrika, stevilo_iteracij):
    [u, v] = zaporedje_u(matrika, stevilo_iteracij)
   # seznam_vrednosti = []
    #for k in range(stevilo_iteracij):
    vrednost = np.min(u) / stevilo_iteracij
    vrednost_2 = np.max(v) / stevilo_iteracij
    povprecje =(vrednost + vrednost_2 ) / 2
       # seznam_vrednosti.append(povprecje)
    #with open("iteracija1_matrika1_15000.txt", 'w', encoding='utf-8') as izhodna:
      #  izhodna.write("{}\n".format(seznam_vrednosti))
    return(vrednost, vrednost_2, povprecje)

 
def analiza1(matrika, stevilo_iteracij):
    korak = 1
    seznam_vrednosti = []
    while korak < stevilo_iteracij:
        seznam_vrednosti.append(vrednost_igre(matrika, korak)[2])
        korak += 1
        
    with open("iteracija1_matrika1_15000_1.txt", 'w', encoding='utf-8') as izhodna:
            izhodna.write("{}\n".format(seznam_vrednosti))
    return(seznam_vrednosti)


#2. iterativni algoritem, ki vrne vrednost matrice igre in Nashevo 
# ravnovesje v obliki vektorja Z = [x_1, ...., x_n, y_1, ..., y_m, v]

#generiramo vektor Z, ki ustreza dolocenim pogojem:

def generiraj_zacetni_vektor(m,n, matrika):
    x_0 = np.random.random(m)
    x_0 /= x_0.sum()
    y_0 = np.random.random(n)
    y_0 /= y_0.sum()
    x_0 = [1 / m ] * m
    y_0 = [1 / n] * n
    v = abs(np.random.randint(low=np.min(matrika), high=np.max(matrika), size=1, dtype=int))
    #x_0 = x_0.tolist()
    #y_0 = y_0.tolist()
    v = v.tolist()
    Z = [*x_0, *y_0, *v]
    return(Z)

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

def iteracija2(stevilo_iteracij, matrika):
    m = matrika.shape[0]
    n = matrika.shape[1]
    #vrednosti = []
    Z = np.array(generiraj_zacetni_vektor(m, n, matrika))
    korak = 0
    razlika = 10
    
    while korak < (stevilo_iteracij):
        prejsni_Z = Z
        produkt_a_z = []
       # vrednosti.append(Z[-1])
        for i in range(m + n):
            A_i = np.array(sestavi_vektor_A_i(m,n,i,matrika))
            a_i_krat_z = np.dot(A_i, Z)
            produkt_a_z.append(a_i_krat_z)
        if all(elem >= 0 for elem in produkt_a_z):
            break
        else:
            produkt_a_z = np.array(produkt_a_z)
            j_k = np.argmin(produkt_a_z)
            if j_k < (n ):
                A_0_j_k = np.array(sestavi_vektor_A_0_i(m, n, j_k))
                A_j_k = np.array(sestavi_vektor_A_i(m, n, j_k, matrika))
                b_0_j_k = np.real(1 / sqrt(np.dot(A_0_j_k, A_0_j_k)))
                B_j_k= np.real(A_j_k / sqrt(np.dot(A_j_k, A_j_k)))
                B_0_j_k = np.real(A_0_j_k / sqrt(np.dot(A_0_j_k, A_0_j_k)))
                cos_theta_jk = np.real((np.dot(A_0_j_k, A_j_k)) / sqrt(np.dot(A_0_j_k, A_0_j_k)* np.dot(A_j_k, A_j_k)))
                Z1 = np.array(Z)
                alfa = np.dot(-Z1, B_j_k) / ((1 - cos_theta_jk ** 2))
                beta = b_0_j_k - np.dot((Z1 + alfa * B_j_k),B_0_j_k)
                zacasni_Z1  = Z1 + alfa * B_j_k + beta * B_0_j_k
               # b_j_k = 1 / sqrt(np.dot(A_j_k, A_j_k))
            
                zacasni_Z = zacasni_Z1.tolist()
                zacasni_Z = zacasni_Z1.tolist()
                zacasni_x = zacasni_Z[0:m]
                zacasni_y = zacasni_Z[m:(m+n)]
                zacasni_v = zacasni_Z[-1]
            
                if all(elem >= 0 for elem in zacasni_Z[0:m]):
                    Z = zacasni_Z
                    korak = korak + 1
                    
                else:
                    korak = korak + 1
                    i = 0
                    seznam_indeksov_negativnih_komponent_x = []
                    seznam_indeksov_nenegativnih_komponent_x = []
                    for komponente_x in zacasni_Z[0:m]:
                        if komponente_x < 0:
                            Z[i] = 0
                            seznam_indeksov_negativnih_komponent_x.append(i)
                            i = i + 1
                        else:
                            seznam_indeksov_nenegativnih_komponent_x.append(i)
                            i = i + 1
                
                    r_plus_s = len(seznam_indeksov_negativnih_komponent_x)
                    r = r_plus_s
                    s = 0
                    while r != 0:
                        vsota_neg_komponent_x_i = 0
                        zacasni_seznam_i_neg_komponent = []
                        zacasni_seznam_i_neneg_komponent = []
                        for i in seznam_indeksov_negativnih_komponent_x:
                            vsota_neg_komponent_x_i = vsota_neg_komponent_x_i + zacasni_Z[i]
                        for i in seznam_indeksov_nenegativnih_komponent_x:
                            vsota = 0
                            x_i = zacasni_Z[i]
                            vsota = x_i + vsota_neg_komponent_x_i / (m - r_plus_s)
                            if vsota < 0:
                                Z[i] = 0
                                zacasni_seznam_i_neg_komponent.append(i)
                                seznam_indeksov_negativnih_komponent_x.append(i)
                            else:
                                zacasni_seznam_i_neneg_komponent.append(i)
                       
                        r = len(zacasni_seznam_i_neg_komponent)
                        r_plus_s = len(seznam_indeksov_negativnih_komponent_x)
                        seznam_indeksov_nenegativnih_komponent_x = zacasni_seznam_i_neneg_komponent
                   
                    vsota_nicelnih_x = 0
                    for i in seznam_indeksov_negativnih_komponent_x:
                        vsota_nicelnih_x = vsota_nicelnih_x + zacasni_Z[i]
                    for i in range(m):
                        if i not in seznam_indeksov_negativnih_komponent_x:
                            Z[i] = zacasni_x[i] + vsota_nicelnih_x / (m - r_plus_s)
                    Z[-1] = zacasni_v
                vsota1 = sum(Z[0:m])
               
                
                
            else:
                A_0_j_k = np.array(sestavi_vektor_A_0_i(m, n, j_k))
                A_j_k = np.array(sestavi_vektor_A_i(m, n, j_k, matrika))
    
                b_0_j_k = np.real(1 / sqrt(np.dot(A_0_j_k, A_0_j_k)))
                B_j_k= np.real(A_j_k / sqrt(np.dot(A_j_k, A_j_k)))
                B_0_j_k = np.real(A_0_j_k / sqrt(np.dot(A_0_j_k, A_0_j_k)))
                cos_theta_jk = np.real((np.dot(A_0_j_k, A_j_k)) / sqrt(np.dot(A_0_j_k, A_0_j_k)* np.dot(A_j_k, A_j_k)))
                Z1 = np.array(Z)
                alfa = np.dot(-Z1, B_j_k) / ((1 - cos_theta_jk ** 2))
                beta = b_0_j_k - np.dot((Z1 + alfa * B_j_k),B_0_j_k)
                zacasni_Z1  = Z1 + alfa * B_j_k + beta * B_0_j_k
                zacasni_Z = zacasni_Z1.tolist()
                zacasni_x = zacasni_Z[0:m]
                zacasni_y = zacasni_Z[m:(m+n)]
                zacasni_v = zacasni_Z[-1]
 
                if all(elem >= 0 for elem in zacasni_Z[m:(m+n)]):
                    Z = zacasni_Z
                    korak = korak + 1
                    
                else:
                    korak = korak + 1
                    i = m
                    seznam_indeksov_negativnih_komponent_x = []
                    seznam_indeksov_nenegativnih_komponent_x = []
                    for komponente_x in zacasni_Z[m:(m+n)]:
                        if komponente_x < 0:
                            Z[i] = 0
                            seznam_indeksov_negativnih_komponent_x.append(i)
                            i = i + 1
                        else:
                            seznam_indeksov_nenegativnih_komponent_x.append(i)
                            i = i + 1
                
                    r_plus_s = len(seznam_indeksov_negativnih_komponent_x)
                    s = 0
                    r = r_plus_s
                    while r != 0:
                        vsota_neg_komponent_x_i = 0
                        zacasni_seznam_i_neg_komponent = []
                        zacasni_seznam_i_neneg_komponent = []
                        for i in seznam_indeksov_negativnih_komponent_x:
                            vsota_neg_komponent_x_i = vsota_neg_komponent_x_i + zacasni_Z[i]
                        for i in seznam_indeksov_nenegativnih_komponent_x:
                            vsota = 0
                            x_i = zacasni_Z[i]
                            vsota = x_i + vsota_neg_komponent_x_i / (n - r_plus_s)
                            if vsota < 0:
                                Z[i] = 0
                                zacasni_seznam_i_neg_komponent.append(i)
                                seznam_indeksov_negativnih_komponent_x.append(i)
                            else:
                                zacasni_seznam_i_neneg_komponent.append(i)
                        seznam_preostalih_komponent = []
                        for i in range(m, n+m):
                            if i not in zacasni_seznam_i_neneg_komponent:
                                if i not in zacasni_seznam_i_neg_komponent:
                                    seznam_preostalih_komponent.append(i)
                        r = len(seznam_indeksov_negativnih_komponent_x)
                        seznam_indeksov_nenegativnih_komponent_x = zacasni_seznam_i_neneg_komponent
                        r_plus_s = len(seznam_indeksov_negativnih_komponent_x)
                    vsota_nicelnih_x = 0
                    for i in seznam_indeksov_negativnih_komponent_x:
                        vsota_nicelnih_x = vsota_nicelnih_x + zacasni_Z[i]
                    for i in range(m,(m+n)):
                        if i not in seznam_indeksov_negativnih_komponent_x:
                            Z[i] = zacasni_Z[i] + vsota_nicelnih_x / (m - r_plus_s)
                    Z[-1] = zacasni_v
                vsota1 = sum(Z[m:(m+n)])
               
               

    with open("iteracija2_matrika1_15000.txt", 'w', encoding='utf-8') as izhodna:
        izhodna.write("{}\n".format(vrednosti))
    
    return(Z)   
## 3. metoda, vrne X, Y in vrednost igre -brownova statistična metoda

def iteracija_brown(matrika, stevilo_iteracij):
    m = matrika.shape[0]
    n = matrika.shape[1]
    P1 = []
    P2 = []
    P11 = np.zeros(m)
    P11[0] = 1
    i = 0
    P1.append(P11)
    A_vektor = np.zeros(n)
    B_vektor = np.zeros(m)
    v_vektor1 = []
    V_vektor2 = []

    for korak in range(stevilo_iteracij):
        P22 = np.zeros(n)
        P11 = np.zeros(m)
        A = matrika[i, 0:(n+1)]
        A_vektor += A
        v1 = min(A_vektor) / (stevilo_iteracij + 1)
        v_vektor1.append(v1)
        j = np.argmin(A_vektor)
        P22[j] = 1
        B = matrika[0:(m+1),j]
        B_vektor += B
        v2 = max(B_vektor) / (stevilo_iteracij +1)
        V_vektor2.append(v2)
        i = np.argmax(B_vektor)
        P11[i] = 1
        P1.append(P11)
        P2.append(P22)
        
    vred1 = max(v_vektor1)
    vred2 = min(V_vektor2)
    vsota1 = 0
    vsota2 = 0 
    for el in P1:
        vsota1 += el
    for el in P2:
        vsota2 += el
    P = vsota1 / (stevilo_iteracij + 1)
    Q = vsota2 / (stevilo_iteracij + 1)
    vred = (vred1 + vred2) / 2

    return(P,Q, vred, vred1, vred2)



#funckiji za merjenje časa
def zmeri_cas(matrika, stevilo_iteracij, algoritem):
    vrednosti = []
    casi = []
    if algoritem == 1:
        for ponovitve in range(10):
            start1 = time.time()
            vrednost = vrednost_igre(matrika, stevilo_iteracij)[-1]
            end1 = time.time()
            cas = end1 - start1
            vrednosti.append(vrednost)
            casi.append(cas)
    else:
        for ponovitve in range(10):
            start2 = time.time()
            vrednost = iteracija2(stevilo_iteracij, matrika)[-1]
            end2 = time.time()
            cas = (end2 - start2)
            vrednosti.append(vrednost)
            casi.append(cas)
            #print(vrednost)
  
    vred = mean(vrednosti)

    t = mean(casi)
    return(t, vred)



#print(zmeri_cas(matrika11,1000,1))
#print(zmeri_cas(matrika, 1,2))
#print(zmeri_cas(matrika, 5<00, 2))
#print(iteracija_brown(matrika, 15000))
#print(iteracija2(100, matrika))
