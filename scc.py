import numpy as np
from scipy.linalg import toeplitz
from time import process_time

def scc_iterativo(x, h):
    
    # Longitudes de las señales x[n] y h[n]
    N1 = len(x)
    N2 = len(h)
    N = N1 + N2 - 1 # Longitud de la señal de salida y[n]

    # Zero-padding de las señales x[n] y h[n] (UTILIZAR NUMPY)  
    x = np.pad(x, (0, N - N1), mode='constant') #Los 0 que faltan en N1 para que sean N
    h = np.pad(h, (0, N - N2), mode='constant') #Los 0 que faltan en N2 para que sean N

    y = np.zeros(N) # Inicializar la señal de salida y[n] con ceros

    H = toeplitz(h, h)  #Primer Parametro es la primera columna y el segundo la primera fila
    
    Ht = np.transpose(H) # Transponer la matriz H para obtener Ht


    for  i in range(N):
        for j in range(N):
            y[i] += x[j] * Ht[i][j]

    return y


N = [2**i for i in range(1, 14)] # Lista de longitudes de señales a probar [2, 4, 8, 16, ..., 8192]

for n in N:
    x = np.random.uniform(-1, 1, n)
    h = np.random.uniform(-1, 1, n)

    s = process_time()
    y = scc_iterativo(x, h)
    f = process_time()

    print(f"N: {n}, Tiempo de ejecución: {f - s} segundos")
