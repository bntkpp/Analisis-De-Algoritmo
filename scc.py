import numpy as np
from scipy.linalg import toeplitz
from time import process_time
import matplotlib.pyplot as plt

def scc_iterativo(x, h):
    
    # Longitudes de las señales x[n] y h[n]
    N1 = len(x)
    N2 = len(h)
    N = N1 + N2 - 1 # Longitud de la señal de salida y[n]

    # Zero-padding de las señales x[n] y h[n] (UTILIZAR NUMPY)  
    x = np.pad(x, (0, N - N1), mode='constant') # Los 0 que faltan en N1 para que sean N
    h = np.pad(h, (0, N - N2), mode='constant') # Los 0 que faltan en N2 para que sean N

    y = np.zeros(N) # Inicializar la señal de salida y[n] con ceros

    H = toeplitz(h, h)  # Primer Parametro es la primera columna y el segundo la primera fila
    
    Ht = np.transpose(H) # Transponer la matriz H para obtener Ht


    for  i in range(N):
        for j in range(N):
            y[i] += x[j] * Ht[i][j]

    return y

def fft(x):
    N = len(x)

    if N == 1: # Caso base
        return x
    
    # Dividir en subproblemas de tamaño N/2
    pares = fft(x[::2]) # Elementos en posiciones pares (x[0], x[2], x[4], ...)
    impares = fft(x[1::2]) # Elementos en posiciones impares (x[1], x[3], x[5], ...)
    
    W = [np.exp(-2j * np.pi * k / N) for k in range(N // 2)] # Calcular los factores de la raíz N-ésima de la unidad

    X = [pares[k] + W[k] * impares[k] for k in range(N // 2)] + [pares[k] - W[k] * impares[k] for k in range(N // 2)]  #Combina la mitad inferior y superior de la FFT

    return np.array(X) # Devuelve la FFT como un array de numpy (Para poder hacer operaciones con ella)
    
def ifft(X):
    N = len(X)

    if N == 1:
        return X
    
    # Dividir en subproblemas de tamaño N/2
    pares = ifft(X[::2]) # Elementos en posiciones pares (x[0], x[2], x[4], ...)
    impares = ifft(X[1::2]) # Elementos en posiciones impares (x[1], x[3], x[5], ...)

    # Dividir en subproblemas de tamaño N/2
    W = [np.exp(-2j * np.pi * k / N) for k in range(N // 2)] # Calcular los factores de la raíz N-ésima de la unidad

    x = [pares[k] + W[k] * impares[k] for k in range(N // 2)] + [pares[k] - W[k] * impares[k] for k in range(N // 2)]  

    return np.array(x) / 2
        
        
def scc_fft(x, h):
    N1 = len(x)
    N2 = len(h)
    N = N1 + N2 - 1 # Longitud de la señal de salida y[n]
    N_potencia2 = 1 << (N - 1).bit_length()  # Siguiente potencia de 2

    x = np.pad(x, (0, N_potencia2 - N1), mode='constant')
    h = np.pad(h, (0, N_potencia2 - N2), mode='constant')

    X = fft(x) # Transformada de Fourier de x[n]
    H = fft(h) # Transformada de Fourier de h[n]
    Y = X * H # Multiplicación de las transformadas de Fourier
    y = ifft(Y) # Transformada inversa de Fourier

    return np.real_if_close(y[:N]) # Devuelve la parte real de la señal de salida y[n] (ya que es real)


tiempo_iterativo = [] # Lista para almacenar los tiempos de ejecución del método iterativo
tiempo_fft = [] # Lista para almacenar los tiempos de ejecución del método FFT
N = [2**i for i in range(1, 14)] # Lista de longitudes de señales a probar [2, 4, 8, 16, ..., 8192]


for n in N:
    x = np.random.uniform(-1, 1, n) # Generar una señal aleatoria x[n]
    h = np.random.uniform(-1, 1, n) # Generar una señal aleatoria h[n]

    s = process_time()
    y1 = scc_iterativo(x, h)
    f = process_time()
    tiempo_iterativo.append(f - s) # Guardar el tiempo de ejecución del método iterativo

    s = process_time()
    y2 = scc_fft(x, h)
    f = process_time()
    tiempo_fft.append(f - s) # Guardar el tiempo de ejecución del método FFT



# Graficar Iterativo vs FFT
plt.plot(N, tiempo_iterativo, label='Iterativo', marker='o')
plt.plot(N, tiempo_fft, label='FFT', marker='x')
plt.xlabel("Tamaño N")
plt.ylabel("Tiempo (s)")
plt.title("Comparación de tiempos de ejecución: SCC Iterativo vs FFT")
plt.legend()
plt.grid(True)
plt.show()
