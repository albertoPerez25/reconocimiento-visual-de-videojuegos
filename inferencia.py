import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt


# 1. Configuración (DEBE coincidir con tu entrenamiento en main.py)
IMG_HEIGHT = 480
IMG_WIDTH = 854
MODEL_PATH = 'best_transfer.keras'

# Las clases del dataset EuroSAT (en orden alfabético, como lo hace flow_from_directory)
CLASS_NAMES = ['GOD_OF_WAR_1', 'HADES', 'HOLLOW_KNIGHT', 'MARIO_GALAXY', 'MINECRAFT', 'UNDERTALE']

def predict_image(image_path):
    # Verificar que la imagen existe
    if not os.path.exists(image_path):
        print(f"Error: La imagen '{image_path}' no existe.")
        return

    print(f"Cargando modelo desde: {MODEL_PATH}...")
    try:
        # 2. Cargar el modelo entrenado
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return

    print("Procesando imagen...")
    
    # 3. Cargar y preprocesar la imagen
    # target_size se encarga de redimensionar la imagen a 854x480 automáticamente
    img = image.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    
    # Convertir a array de numpy
    img_array = image.img_to_array(img)
    
    # EfficientNetV2 maneja la normalización internamente (0-255), 
    # pero necesitamos añadir la dimensión del batch.
    # La forma pasa de (854, 480, 3) a (1, 854, 480, 3)
    img_array = tf.expand_dims(img_array, 0) 

    # 4. Realizar la predicción
    predictions = model.predict(img_array)
    
    # Como tu última capa es softmax, 'predictions' ya contiene probabilidades
    score = tf.nn.softmax(predictions[0]) # Opcional, para asegurar formato
    
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_name = CLASS_NAMES[predicted_class_index]
    confidence = 100 * np.max(predictions[0])

    # 5. Mostrar resultados
    print("-" * 30)
    print(f"RESULTADO DE LA INFERENCIA")
    print("-" * 30)
    print(f"Imagen: {image_path}")
    print(f"Predicción: {predicted_class_name}")
    print(f"Confianza: {confidence:.2f}%")
    print("-" * 30)

    # Opcional: Mostrar la imagen si estás en un entorno gráfico (Jupyter/Local)
    plt.imshow(img)
    plt.title(f"Pred: {predicted_class_name} ({confidence:.1f}%)")
    plt.axis("off")
    plt.show()

# --- EJECUCIÓN ---
# Cambia 'ruta_de_tu_imagen.jpg' por la imagen que quieras probar
# Ejemplo:
# predict_image('test_images/forest_sample.jpg')

if __name__ == "__main__":
    # Puedes pedir la ruta por consola para probar rápido
    path = input("Introduce la ruta de la imagen a clasificar: ")
    predict_image(path)