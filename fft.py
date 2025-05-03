import numpy as np
from scipy.linalg import toeplitz
from time import process_time


def fft(x):
    N = len(x)
    if N == 1:
        return x
    else:
        # Dividir en subproblemas de tamaño N/2
        pares = fft(x[::2]) # Elementos en posiciones pares (x[0], x[2], x[4], ...)
        impares = fft(x[1::2]) # Elementos en posiciones impares (x[1], x[3], x[5], ...)

        X = np.zeros(N, dtype=complex) # Inicializar la salida con ceros complejos

        for k in range(N // 2):
            W = (np.exp(-2j * np.pi * k / N))
            X[k] = pares[k] + W * impares[k] 
            X[k + N // 2] = pares[k] - W * impares[k] 

        return X
        
    

x = [1, 2, 3, 4]
X1 = fft(x)              # tu implementación recursiva
X2 = np.fft.fft(x)       # NumPy
print(X1)
print(X2)
print(np.allclose(X1, X2))  