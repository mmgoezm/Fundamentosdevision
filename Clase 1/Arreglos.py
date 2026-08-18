# IMPORTACIÓN DE LIBRERÍAS FUNDAMENTALES
import numpy as np

# ==============================================================================
# SECCIÓN 1: CREACIÓN BÁSICA DE MATRICES (ARRAYS)
# ==============================================================================

print("--- 1. CREACIÓN BÁSICA ---")
matriz_1d = np.array([1, 2, 3, 4, 5])
matriz_2d = np.array([[1, 2, 3], [4, 5, 6]])

print("Matriz 1D:\n", matriz_1d)
print("Matriz 2D:\n", matriz_2d, "\n")

# ==============================================================================
# SECCIÓN 2: CREACIÓN DE MATRICES ESPECIALES
# ==============================================================================

print("--- 2. MATRICES ESPECIALES ---")

matriz_ceros = np.zeros((2, 3))
matriz_unos = np.ones((3, 2))
matriz_identidad = np.eye(3)

print("Matriz de Ceros (2x3):\n", matriz_ceros)
print("Matriz de Unos (2x3):\n", matriz_unos)
print("Matriz Identidad (3x3):\n", matriz_identidad, "\n")

# ==============================================================================
# SECCIÓN 3: GENERACIÓN DE SECUENCIAS
# ==============================================================================
print("--- 3. SECUENCIAS ---")
secuencia_pasos = np.arange(0, 11, 2) # De 0 a 10 (exclusivo) con saltos de 2
secuencia_lineal = np.linspace(0, 1, 5) # 5 números equiespaciados entre 0 y 1
print("Arange (saltos de 2):", secuencia_pasos)
print("Linspace (5 valores):", secuencia_lineal, "\n")

# ==============================================================================
# SECCIÓN 4: MATRICES CON VALORES ALEATORIOS
# ==============================================================================

print("--- 4. VALORES ALEATORIOS ---")
matriz_aleatoria = np.random.rand(2, 2) # Valores entre 0 y 1
enteros_aleatorios = np.random.randint(10, 50, (2, 3)) # Enteros entre 10 y 49
print("Matriz aleatoria uniforme:\n", matriz_aleatoria)
print("Enteros aleatorios:\n", enteros_aleatorios, "\n")

# ==============================================================================
# SECCIÓN 5: ATRIBUTOS DE UNA MATRIZ
# ==============================================================================

print("--- 5. ATRIBUTOS DE MATRIZ ---")
ejemplo = np.array([[1.5, 2.5], [3.5, 4.5], [5.5, 6.5]])
print("Matriz de ejemplo:\n", ejemplo)
print("Forma (shape):", ejemplo.shape)
print("Número de dimensiones (ndim):", ejemplo.ndim)
print("Cantidad de elementos (size):", ejemplo.size)
print("Tipo de dato (dtype):", ejemplo.dtype, "\n")

# ==============================================================================
# SECCIÓN 6: INDEXACIÓN BÁSICA
# ==============================================================================

print("--- 6. INDEXACIÓN BÁSICA ---")
mat = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])
print("Matriz original:\n", mat)
print("Elemento en fila 1, col 2:", mat[1, 2]) # Fila índice 1, Col índice 2 (es el 60)
print("Toda la primera fila:", mat[0, :], "\n")

# ==============================================================================
# SECCIÓN 7: SLICING (REBANADO)
# ==============================================================================

print("--- 7. SLICING ---")
print("Matriz original:\n", mat)
sub_matriz = mat[0:2, 1:3] # Filas 0 y 1, Columnas 1 y 2
print("Sub-matriz (esquina superior derecha):\n", sub_matriz, "\n")

# ==============================================================================
# SECCIÓN 8: INDEXACIÓN BOOLEANA (FILTRADO)
# ==============================================================================

print("--- 8. INDEXACIÓN BOOLEANA ---")
datos = np.array([15, 2, 30, 4, 55, 6])
print("Datos originales:", datos)
condicion = datos > 10
print("Valores mayores a 10:", datos[condicion], "\n")

# ==============================================================================
# SECCIÓN 9: REDIMENSIONAMIENTO (RESHAPE)
# ==============================================================================

print("--- 9. REDIMENSIONAMIENTO (RESHAPE) ---")
array_1d = np.arange(12)
print("Array 1D original:", array_1d)
array_2d = array_1d.reshape((3, 4)) # Convierte 1D de 12 elementos en 3x4
print("Matriz 3x4 (Reshape):\n", array_2d, "\n")

# ==============================================================================
# SECCIÓN 10: APLANAR MATRICES
# ==============================================================================

print("--- 10. APLANAR MATRICES ---")
print("Matriz 2D original:\n", array_2d)
aplanada = array_2d.flatten() # Devuelve una copia 1D
print("Matriz aplanada (Flatten):", aplanada, "\n")

# ==============================================================================
# SECCIÓN 11: TRANSPOSICIÓN
# ==============================================================================

print("--- 11. TRANSPOSICIÓN ---")
mat_T = np.array([[1, 2], [3, 4], [5, 6]])
print("Original (3x2):\n", mat_T)
print("Transpuesta (2x3):\n", mat_T.T, "\n")

# ==============================================================================
# SECCIÓN 12: OPERACIONES ARITMÉTICAS ELEMENTO A ELEMENTO
# ==============================================================================

print("--- 12. ARITMÉTICA ELEMENTO A ELEMENTO ---")
A = np.array([[1, 2], [3, 4]])
B = np.array([[10, 10], [10, 10]])
print("Suma:\n", A + B)
print("Multiplicación:\n", A * B) # Ojo, esto NO es producto matricial
print("Exponente:\n", A ** 2, "\n")

# ==============================================================================
# SECCIÓN 13: PRODUCTO MATRICIAL (DOT PRODUCT)
# ==============================================================================

print("--- 13. PRODUCTO MATRICIAL ---")
C = np.array([[1, 2], [3, 4]])
D = np.array([[2, 0], [1, 2]])
producto_punto = np.dot(C, D)
producto_arroba = C @ D # Sintaxis moderna para producto matricial
print("Producto matricial:\n", producto_arroba, "\n")

# ==============================================================================
# SECCIÓN 14: ESTADÍSTICA BÁSICA
# ==============================================================================

print("--- 14. ESTADÍSTICA ---")
estadisticas = np.array([[1, 5, 3], [7, 2, 9]])
print("Matriz:\n", estadisticas)
print("Suma total:", np.sum(estadisticas))
print("Promedio (Media):", np.mean(estadisticas))
print("Valor Máximo:", np.max(estadisticas), "\n")

# ==============================================================================
# SECCIÓN 15: OPERACIONES POR EJES (AXIS)
# ==============================================================================

print("--- 15. OPERACIONES POR EJES ---")
print("Matriz original:\n", estadisticas)
print("Suma por columnas (axis=0):", np.sum(estadisticas, axis=0))
print("Suma por filas (axis=1):", np.sum(estadisticas, axis=1), "\n")

# ==============================================================================
# SECCIÓN 16: BROADCASTING
# ==============================================================================

print("--- 16. BROADCASTING ---")
base = np.array([[1, 2, 3], [4, 5, 6]])
vector = np.array([10, 100, 1000])
print("Matriz:\n", base)
print("Vector:", vector)
print("Suma (Broadcasting de vector a cada fila):\n", base + vector, "\n")

# ==============================================================================
# SECCIÓN 17: CONCATENACIÓN Y APILAMIENTO
# ==============================================================================

print("--- 17. CONCATENACIÓN ---")
x = np.array([[1, 2]])
y = np.array([[3, 4]])
print("Apilamiento vertical (Vstack):\n", np.vstack((x, y)))
print("Apilamiento horizontal (Hstack):\n", np.hstack((x, y)), "\n")

# ==============================================================================
# SECCIÓN 18: DIVISIÓN DE MATRICES (SPLIT)
# ==============================================================================

print("--- 18. DIVISIÓN DE MATRICES ---")
matriz_a_dividir = np.arange(16).reshape(4, 4)
print("Matriz original:\n", matriz_a_dividir)
# Divide en 2 mitades horizontalmente (por la mitad de las filas)
mitades = np.vsplit(matriz_a_dividir, 2)
print("Primera mitad:\n", mitades[0])
print("Segunda mitad:\n", mitades[1], "\n")

# ==============================================================================
# SECCIÓN 19: AGREGAR Y ELIMINAR ELEMENTOS
# ==============================================================================

print("--- 19. AGREGAR / ELIMINAR ---")
arr = np.array([10, 20, 30])
print("Original:", arr)
arr_agregado = np.append(arr, [40, 50])
print("Después de Append:", arr_agregado)
arr_eliminado = np.delete(arr_agregado, 1) # Elimina el índice 1 (el valor 20)
print("Después de Delete (índice 1):", arr_eliminado, "\n")

# ==============================================================================
# SECCIÓN 20: VALORES ÚNICOS Y CONTEOS
# ==============================================================================

print("--- 20. VALORES ÚNICOS ---")
repetidos = np.array([1, 2, 2, 3, 1, 4, 2, 5, 3])
print("Arreglo con repetidos:", repetidos)
unicos, conteos = np.unique(repetidos, return_counts=True)
print("Valores únicos:", unicos)
print("Frecuencia de cada valor:", conteos)

