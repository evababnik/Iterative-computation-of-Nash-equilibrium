from cmath import sqrt
from statistics import mean
import numpy as np
import math
import cmath
import scipy
from scipy.optimize import minimize
from sympy import Nor
import time
from scipy.optimize import linprog

#1. iterativni algoritem, ki vrne vrednost matrične igre in optimalno strategijo 
# v obliki vektorja Z = [x_1, ...., x_n, y_1, ..., y_m, v]

#generiramo vektor Z, ki ustreza dolocenim pogojem:

def generiraj_zacetni_vektor(m,n, matrika):
    x_0 = np.random.random(m)
    x_0 /= x_0.sum()
    y_0 = np.random.random(n)
    y_0 /= y_0.sum()
    x_0 = [1 / m ] * m
    y_0 = [1 / n] * n
    v = abs(np.random.randint(low=np.min(matrika), high=np.max(matrika), size=1, dtype=int))
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

def iteracija1(stevilo_iteracij, matrika):
    pravi_Z = np.array(resitev_LP(matrika))
    
    m = matrika.shape[0]
    n = matrika.shape[1]
    Z = np.array(generiraj_zacetni_vektor(m, n, matrika))
    korak = 0
    napaka = []
    vrednosti = []
    cas = []
    start = time.time()
    while korak < (stevilo_iteracij):
       
        vrednosti.append(Z[-1])
        napaka.append(max(abs(pravi_Z - Z))) 
        
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
                zacasni_Z = zacasni_Z1.tolist()
                zacasni_x = zacasni_Z[0:m]
                zacasni_y = zacasni_Z[m:(m+n)]
                zacasni_v = zacasni_Z[-1]
                if all(elem >= 0 for elem in zacasni_Z[0:m]):
                    Z = zacasni_Z
                    korak = korak + 1
                    end = time.time()
                    cas.append(end-start)
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
                    end = time.time()
                    cas.append(end-start)
               
                
                
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
                    end = time.time()
                    cas.append(end-start)
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
                    end = time.time()
                    cas.append(end-start)
                
    #with open("iteracija1_matrika0_napake.txt", 'w', encoding='utf-8') as izhodna:
       # izhodna.write("{}\n".format(napaka))         
    #with open("iteracija1_matrika0_100_vrednosti.txt", 'w', encoding='utf-8') as izhodna:
        #izhodna.write("{}\n".format(vrednosti))
    with open("iteracija1_matrika0_100_cas.txt", 'w', encoding='utf-8') as izhodna:
        izhodna.write("{}\n".format(cas))
    return(Z)  

## 2. metoda, vrne X, Y in vrednost igre - Brownova statistična metoda

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

    return([P,Q, vred, vred1, vred2])


#linearni program za računanje rešitve matrične igre

def resitev_LP(matrika):
    n = matrika.shape[0]
    m = matrika.shape[1]
    C = [1] * n # coefficients of the linear objective function vector
    A = []  #inequality constraints matrix
    for j in range(m):
        A_j = matrika[0:n, j] 
        A_j = A_j.tolist()
        A_j =  [element * (-1) for element in A_j]
        A.append(A_j)
    
    for i in range(n):
        nicle = [0] * n
        nicle[i] = -1
        A.append(nicle)

    A = np.array(A)
    B1 = [-1] * m
    B2 = [0] * n
    B = np.array([*B1, *B2])   #inequality constraints vector

    resitev = linprog(C, A_ub=A, b_ub=B)
    p_i = resitev.x
    X = p_i * (1 / resitev.fun)

    CY = [-1] * m
    AY = []
    for i in range(n):
        A_i = matrika[i, 0:m]
        A_i = A_i.tolist()
        AY.append(A_i)

    BY = [1] * n
    meje = [(0,1)]
    resitev2 = linprog(c = CY, A_ub = AY, b_ub = BY, bounds=meje)
    
    q_i = resitev2.x
    Y = q_i * (1 / resitev2.fun) * (-1)
    vrednost = 1 / sum(resitev.x)
    Y = Y.tolist()
    X = X.tolist()
    Z = [*X, *Y, vrednost]
    return(Z)
        
### Analiza konvergence

def analiza2(matrika, stevilo_iteracij):
    korak = 1
    seznam_povp_vrednosti = []
    seznam_sp_vrednosti = []
    seznam_zg_vrednosti = []
    pravi_z = np.array(resitev_LP(matrika))
    napaka = []
    cas = [0]
    while max(cas) < 0.5:
        cas_vmesni = []
        X_vmesni = []
        Y_vmesni = []
        povp_vr_vmesni = []
        sp_vr_vmesni = []
        zg_vr_vmesni = []
        
        start = time.time()
        resitev = np.array(iteracija_brown(matrika, korak))
        end = time.time()
        cas_vmesni.append(end-start)
        X_vmesni.append(resitev[0].tolist())
        Y_vmesni.append(resitev[1].tolist())
        
        povp_vr_vmesni.append(resitev[2])
        sp_vr_vmesni.append(resitev[3])
        zg_vr_vmesni.append(resitev[4])
        
        cas.append(mean(cas_vmesni))
        X = [sum(x)/len(x) for x in zip(*X_vmesni)]
        Y = [sum(x)/len(x) for x in zip(*Y_vmesni)]
        
        povp_vr = mean(povp_vr_vmesni)
        sp_vr = mean(sp_vr_vmesni)
        zg_vr = mean(zg_vr_vmesni)
        
        Z = np.array([*X, *Y, sp_vr])
        a =max(abs(Z-pravi_Z))
        napaka.append(max(abs(Z-pravi_Z)))
        seznam_povp_vrednosti.append(povp_vr)
        seznam_sp_vrednosti.append(sp_vr)
        seznam_zg_vrednosti.append(zg_vr)
        korak += 1
        
    #with open("iteracija_brown_matrika1_15000_povp.txt", 'w', encoding='utf-8') as izhodna:
           # izhodna.write("{}\n".format(seznam_povp_vrednosti))
    #with open("iteracija_brown_matrika0_100.txt", 'w', encoding='utf-8') as izhodna:
           # izhodna.write("{}\n".format(seznam_sp_vrednosti))
   # with open("iteracija_brown_matrika1_15000_zg.txt", 'w', encoding='utf-8') as izhodna:
           # izhodna.write("{}\n".format(seznam_zg_vrednosti))
    with open("iteracija_brown_matrika0_napaka_0.5.txt", 'w', encoding='utf-8') as izhodna:
        izhodna.write("{}\n".format(napaka))
    with open("iteracija_brown_matrika0_cas_0.5.txt", 'w', encoding='utf-8') as izhodna:
        izhodna.write("{}\n".format(cas))
    return(korak)


matrika2 = np.array([[0, -1, -2, 0, 0, 2, 1], [1, 0, 0, 4, 0, 0, -1], [2, 0, 0 , -1, 1, 0, -1], [0, -4, 1, 0, 0, -1, 1], [0, 0, -1, 0 , 0, -3, 1], [-2, 0, 0, 1, 3, 0, -1], [-1, 1, 1, -1, - 1, 1, 0]])
matrika1= np.array([[4, 3], [2, 4], [5, 2]])


#matrika21 = np.array([[5, 2, 9, 4, 10, 8 , 6],[4, 2, 2, 4, 3, 4 , 7], [3, 2, 3,8, 10, 4 , 2],[5, 3, 3, 4, 10, 4 , 3],[8, 2, 3, 4, 11, 7, 3],[4, 2, 4, 2, 8, 4 , 4],[3, 4, 3, 3, 9, 5 , 3]])

