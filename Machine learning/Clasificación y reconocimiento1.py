"""
======================================================================
==  Codigo de detalle de comportamiento de clasificación basica     ==
==               Universidad ITM                                    ==
==               Curso: Instrodución a vision por compitador        ==
==               Profesor: Manuel Mauricio Goez                     ==
======================================================================
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt # Para crear gráficos
import seaborn as sns # Para gráficos más avanzados (mapas de calor)
from typing import Tuple, List, Dict, Optional # Type organización de tipos
from tqdm import tqdm # Barra de progreso visual

# librerías de machine learning
from sklearn.model_selection import train_test_split # Dividir datos (entrenamiento/prueba)
from sklearn.svm import SVC # Clasificador Support Vector Machine
from sklearn.neural_network import MLPClassifier # Clasificador Red Neuronal (Multi-layer Perceptron)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix # Métricas de evaluación


class SimpleImageVectorizerOpenCV:
    def __init__(self, target_size: Tuple[int, int] = (32, 32)):
        self.target_size = target_size # Tamaño (ancho, alto) para redimensionar
        self.data: List[np.ndarray] = [] # Lista para vectores de características
        self.labels: List[int] = []      # Lista para etiquetas numéricas
        self.class_names: List[str] = [] # Lista para nombres de clases
        self._class_map: Dict[str, int] = {} # Mapeo interno nombre_clase -> etiqueta_numérica
        print(f"Vectorizador inicializado. Tamaño objetivo: {self.target_size}")
    def _process_image(self, image_path: str) -> Optional[np.ndarray]:
        """Procesa una imagen: Carga gris -> Redimensiona -> Aplana -> Normaliza [0,1]."""
        try:
            img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) #
            if img_gray is None: return None
            img_resized = cv2.resize(img_gray, self.target_size, interpolation=cv2.INTER_AREA) # Redimensiona
            return img_resized.flatten() / 255.0 # Aplana a vector 1D y normaliza
        except Exception as e:
            print(f"Error procesando {image_path}: {e}")
            return None # Error durante el proceso

    def load_from_directory(self, image_dir: str) -> None:

        if not os.path.isdir(image_dir): return # Verifica si existe el directorio
        print(f"Cargando desde: {image_dir}")
        self.data, self.labels, self.class_names, self._class_map = [], [], [], {} # Reinicia listas

        # Itera sobre subdirectorios (clases)
        for current_label, class_name in enumerate(sorted(os.listdir(image_dir))):
            class_dir = os.path.join(image_dir, class_name)
            if not os.path.isdir(class_dir): continue
            self._class_map[class_name] = current_label
            self.class_names.append(class_name)

            # Itera sobre imágenes dentro del directorio de clase
            for filename in tqdm(os.listdir(class_dir), desc=f"Procesando {class_name}", unit="img"):
                file_path = os.path.join(class_dir, filename)
                if os.path.isfile(file_path):
                    image_vector = self._process_image(file_path) # Procesa la imagen
                    if image_vector is not None:
                        self.data.append(image_vector) # Guarda el vector
                        self.labels.append(current_label) # Guarda la etiqueta numérica

        # Convierte listas a arrays NumPy
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
        self._print_summary() # Muestra resumen

    def _print_summary(self) -> None:
        """Imprime resumen de datos cargados."""
        print("Carga completada.")
        print(f"  Imágenes procesadas: {len(self.labels)}")
        if len(self.labels) > 0:
            print(f"  Shape datos (X): {self.data.shape}") # Muestra (N_imagenes, N_features)
            print(f"  Shape etiquetas (y): {self.labels.shape}") # Muestra (N_imagenes,)
            print(f"  Clases: {self.class_names}")

    def get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Devuelve los datos (X) y etiquetas (y) como arrays NumPy."""
        return self.data, self.labels

#  Función para Entrenar y Evaluar
def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_name, class_names_list):
    print(f"\n Entrenando {model_name} ")
    model.fit(X_train, y_train) # Entrena el modelo
    print(f"{model_name} entrenado.")
    y_pred = model.predict(X_test) # Predice en el conjunto de prueba
    accuracy = accuracy_score(y_test, y_pred) # Calcula exactitud

    # Muestra resultados
    print(f"\nResultados {model_name}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print("  Reporte de Clasificación:")
    print(classification_report(y_test, y_pred, target_names=class_names_list))

    # Devuelve la exactitud y la matriz de confusión
    return accuracy, confusion_matrix(y_test, y_pred)

# Función para Graficar Resultados
def plot_results(accuracies, confusion_matrices, class_names_list):
    print("\n-Generando Gráficos ")
    try:
        # Gráfico de Barras: Comparación de Accuracy
        plt.figure(figsize=(8, 5))
        models = ['SVM', 'Red Neuronal (MLP)']
        plt.bar(models, accuracies, color=['skyblue', 'lightgreen'])
        plt.ylabel('Accuracy')
        plt.title('Comparación de Accuracy')
        plt.ylim(0, 1.1)
        for i, acc in enumerate(accuracies): plt.text(i, acc + 0.01, f'{acc:.4f}', ha='center')
        plt.tight_layout()

        # Mapas de Calor: Matrices de Confusión
        for i, (name, cm) in enumerate(zip(models, confusion_matrices)):
            plt.figure(figsize=(7, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues' if i == 0 else 'Greens',
                        xticklabels=class_names_list, yticklabels=class_names_list)
            plt.xlabel('Predicción'); plt.ylabel('Valor Real')
            plt.title(f'Matriz de Confusión - {name}')
            plt.tight_layout()

        print("Mostrando gráficos...")
        plt.show() # Muestra todos los gráficos generados
    except ImportError:
        print("Error: Instala matplotlib y seaborn para ver gráficos (`pip install matplotlib seaborn`)")
    except Exception as e:
        print(f"Error al graficar: {e}")

# Ejecución principal

# --- Parámetros Configurables ---
dataset_path = "Dataset4"
vector_size = (92, 92)    # Tamaño para redimensionar imágenes
test_set_proportion = 0.50 # Porcentaje de datos para prue0ba (ej. 0.25 = 25%)
reproducibility_seed = 42  # Semilla para reproducibilidad

# Comprueba si el directorio del dataset existe antes de continuar.
if not os.path.isdir(dataset_path):
    print(f"ERROR: Directorio '{dataset_path}' no encontrado.")

else:

    vectorizer = SimpleImageVectorizerOpenCV(target_size=vector_size)
    vectorizer.load_from_directory(dataset_path)
    X, y = vectorizer.get_data()
    class_names_list = vectorizer.class_names

    if len(X) == 0:
        print("Error: No se cargaron datos.")
    else:
        # División en Conjuntos de Entrenamiento y Prueba
        print("\nDividiendo datos ")
        try:
            # Divide los datos X e y en entrenamiento y prueba.
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_set_proportion, random_state=reproducibility_seed, stratify=y
            )
            print(f"División: {len(X_train)} entreno, {len(X_test)} prueba.")
        except Exception as e:
            print(f"Error en división: {e}. revisar tamaño de conjunto de prueba.")
        else:
            # Definición de Modelos
            svm_model = SVC(random_state=reproducibility_seed) # SVM (por defecto: kernel rbf)
            nn_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=reproducibility_seed, early_stopping=True) # Red Neuronal simple

            # Entrenamiento y Evaluación (Iterativo)
            results = {} # Diccionario para almacenar resultados de cada modelo.
            # Itera sobre cada modelo definido.
            for model, name in [(svm_model, "SVM"), (nn_model, "Red Neuronal (MLP)")]:
                accuracy, cm = train_and_evaluate_model(model, X_train, y_train, X_test, y_test, name, class_names_list)
                results[name] = {'accuracy': accuracy, 'cm': cm}

            #  Visualización de Resultados
            plot_results(
                [res['accuracy'] for res in results.values()], # Lista de accuracies
                [res['cm'] for res in results.values()],       # Lista de matrices de confusión
                class_names_list                               # Lista de nombres de clases
            )

            print("\nProceso Terminado ")



