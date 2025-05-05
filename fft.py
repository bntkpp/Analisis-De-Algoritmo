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
    
def ifft(X):
    N = len(X)
    if N == 1:
        return X
    else:
        # Dividir en subproblemas de tamaño N/2
        pares = ifft(X[::2]) # Elementos en posiciones pares (x[0], x[2], x[4], ...)
        impares = ifft(X[1::2]) # Elementos en posiciones impares (x[1], x[3], x[5], ...)

        x = np.zeros(N, dtype=complex) # Inicializar la salida con ceros complejos

        for k in range(N // 2):
            W = (np.exp(2j * np.pi * k / N))
            x[k] = pares[k] + W * impares[k] 
            x[k + N // 2] = pares[k] - W * impares[k] 

        return x / 2
        
def scc_fft(x, h):
    N1 = len(x)
    N2 = len(h)
    N = N1 + N2 - 1 # Longitud de la señal de salida y[n]
    N_potencia2 = 1 << (N - 1).bit_length()  # siguiente potencia de 2

    x = np.pad(x, (0, N_potencia2 - N1), mode='constant')
    h = np.pad(h, (0, N_potencia2 - N2), mode='constant')

    X = fft(x)
    H = fft(h)
    Y = X * H
    y = ifft(Y)

    
x = [1, 2, 3, 4]
X = fft(x)
print(X)

x_recuperado = ifft(X)
print(x_recuperado)


print("Original:", x)
print("IFFT(FFT):", np.real_if_close(x_recuperado))
print("¿Coincide?:", np.allclose(x, x_recuperado))