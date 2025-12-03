import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import EfficientNetV2B0
from transformers import TFViTModel
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pathlib
import PIL.Image
import time
from sklearn.metrics import classification_report, confusion_matrix
import gc 

# Reproducibilidad
SEED = 2025
tf.random.set_seed(SEED)
np.random.seed(SEED)

print(f"TensorFlow Version: {tf.__version__}")
print(f"GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")


KAGGLE_PATH = '/kaggle/input/eurosat-dataset/EuroSAT'
# Ruta que tendria en local.
LOCAL_PATH = './images_dataset' 

# Kaggle siempre define la variable de entorno 'KAGGLE_KERNEL_RUN_TYPE'
if os.environ.get('KAGGLE_KERNEL_RUN_TYPE') is not None:
    print("Entorno de Kaggle")
    data_dir = KAGGLE_PATH
else:
    data_dir = LOCAL_PATH
    print("Entorno Local:",data_dir)
    
data_dir = pathlib.Path(data_dir)

# Verificación del contenido
all_images = list(data_dir.glob('*/*.jpg'))
image_count = len(all_images)
print(f"Total de imágenes encontradas: {image_count}")

# Verificación de dimensiones reales
first_image = PIL.Image.open(all_images[0])
print(f"Dimensiones reales de una imagen de muestra: {first_image.size}")
print(f"Formato de imagen: {first_image.format}")

# Parámetros Globales
# Ajustamos las constantes al tamaño real detectado (debería ser 64x64)
BATCH_SIZE = 16
IMG_HEIGHT = 480
IMG_WIDTH = 854


# Cargamos el dataset completo sin dividir inicialmente
full_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    seed=SEED,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=True,
    label_mode='int'
)

class_names = full_ds.class_names
print(f"Clases encontradas: {class_names}")

# Calculamos el número de batches
n_batches = tf.data.experimental.cardinality(full_ds).numpy()

train_size = int(0.7 * n_batches)
val_size = int(0.15 * n_batches)
test_size = n_batches - train_size - val_size

print(f"Batches -> Train: {train_size}, Val: {val_size}, Test: {test_size}")

# Realizar la división usando take() y skip()
train_ds = full_ds.take(train_size)
remaining_ds = full_ds.skip(train_size)
val_ds = remaining_ds.take(val_size)
test_ds = remaining_ds.skip(val_size)


AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)




######



def plot_history(history, title="Model History"):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))
    
    # Gráfica de Precisión
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.title(f'{title} - Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True)

    # Gráfica de Pérdida
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.title(f'{title} - Loss')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()


# Usar un modelo que ya sabe ver (pre-entrenado por Google) y adaptarlo.

def build_transfer_model():
    inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    
    x = layers.RandomFlip("horizontal_and_vertical")(inputs)
    x = layers.RandomRotation(0.2)(x)
    x = layers.RandomZoom(0.1)(x)
    
    # EfficientNet ya trae de serie la normalización de pixeles.
    
    base_model = EfficientNetV2B0(
        include_top=False, # Le quitamos la capa final para poner la nuestra final.
        weights='imagenet', # Cargamos los conocimientos previos 
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
    )
    
    # Freezing para que no entrene nada nuevo
    base_model.trainable = False
    
    # Conectamos nuestra entrada al modelo base
    # training=False para que las estadísticas internas del modelo sean con nuestros datos
    x = base_model(x, training=False)
    

    # El modelo base nos devuelve un montón de mapas de características
    # Con esto promediamos todo en un solo vector de 1280 números por imagen.
    x = layers.GlobalAveragePooling2D()(x) 
    x = layers.Dropout(0.2)(x) # Apagamos neuronas al azar para evitar overfitting
    outputs = layers.Dense(10, activation='softmax')(x) # 10 clases finales
    
    model = keras.Model(inputs, outputs, name="Transfer_EfficientNetV2")
    return model, base_model

    

transfer_model, base_model = build_transfer_model()
transfer_model.summary()

transfer_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

callbacks_transfer = [
    # Si no mejora en 5 capas se para
    callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    callbacks.ModelCheckpoint('best_transfer.keras', monitor='val_accuracy', save_best_only=True, verbose=0)
]

print("\nFeature Extraction")
start_time_tl = time.time()

history_tl = transfer_model.fit(
    train_ds,
    epochs=20, # aprende rápido por lo que no necesitamos un numero alto
    validation_data=val_ds,
    callbacks=callbacks_transfer
)

# FineTuning 

# Descongelamos
base_model.trainable = True

# Re-compilamos Low Learning Rate
# Usamos una velocidad de aprendizaje muy pequeña (1e-5).
# Si le ponemos una velocidad mas alta corremos el riesgo de sobreescribir aprendizaje que era correcto
transfer_model.compile(
    optimizer=keras.optimizers.Adam(1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

callbacks_finetune = [
    callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    # Seguimos guardando si superamos el récord
    callbacks.ModelCheckpoint('best_transfer.keras', monitor='val_accuracy', save_best_only=True, verbose=0)
]

print("\nFineTuning")
history_finetune = transfer_model.fit(
    train_ds,
    epochs=20, 
    initial_epoch=history_tl.epoch[-1],
    validation_data=val_ds,
    callbacks=callbacks_finetune
)

end_time_tl = time.time()
tl_training_time = end_time_tl - start_time_tl
print(f"\nTiempo total (Extraction + FineTuning): {tl_training_time:.2f} segundos")

acc = history_tl.history['accuracy'] + history_finetune.history['accuracy']
val_acc = history_tl.history['val_accuracy'] + history_finetune.history['val_accuracy']
loss = history_tl.history['loss'] + history_finetune.history['loss']
val_loss = history_tl.history['val_loss'] + history_finetune.history['val_loss']

# Graficamos
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.plot([len(history_tl.history['accuracy'])-1,len(history_tl.history['accuracy'])-1], 
         plt.ylim(), label='Inicio Fine Tuning', ls='--') 
plt.legend(loc='lower right')
plt.title('Transfer Learning: Evolución de Precisión')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.plot([len(history_tl.history['loss'])-1,len(history_tl.history['loss'])-1], 
         plt.ylim(), label='Inicio Fine Tuning', ls='--')
plt.legend(loc='upper right')
plt.title('Transfer Learning: Evolución de Pérdida')
plt.grid(True)
plt.show()

# Comparativa 

# Intentamos recuperar métricas de los modelos anteriores (si existen en memoria)
try:
    mlp_acc = max(history_mlp.history['val_accuracy'])
    cnn_acc = max(history_cnn.history['val_accuracy'])
    mlp_par = mlp_model.count_params()
    cnn_par = cnn_model.count_params()
except NameError:
    mlp_acc, cnn_acc, mlp_par, cnn_par = 0, 0, 0, 0 # Ponemos ceros si no se corrieron antes

tl_acc = max(val_acc)
tl_par = transfer_model.count_params()

print("\n" + "="*60)
print("      COMPARATIVA FINAL DE ARQUITECTURAS")
print("="*60)
print(f"{'Modelo':<20} | {'Parámetros':<12} | {'Val Acc':<10} | {'Tiempo(s)':<10}")
print("-" * 60)
print(f"{'MLP (Básico)':<20} | {mlp_par:<12,} | {mlp_acc:.4f}     | {mlp_training_time if 'mlp_training_time' in locals() else 0:.1f}")
print(f"{'CNN (Propia)':<20} | {cnn_par:<12,} | {cnn_acc:.4f}     | {cnn_training_time if 'cnn_training_time' in locals() else 0:.1f}")
print(f"{'Transfer Learning':<20} | {tl_par:<12,} | {tl_acc:.4f}     | {tl_training_time:.1f}")
print("-" * 60)