import os

# Configuración de entorno
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import time
import pathlib
import PIL.Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.applications import EfficientNetV2B0


# Reproducibilidad
SEED = 2025
tf.random.set_seed(SEED)
np.random.seed(SEED)

# Constantes de Configuración
IMG_HEIGHT = 480
IMG_WIDTH = 854
BATCH_SIZE = 16

def prepare_dataset():
    """
    Carga, divide y optimiza el dataset de imágenes.
    Detecta automáticamente si se ejecuta en Kaggle o en Local.

    Returns:
        tuple: (train_ds, val_ds, class_names)
            - train_ds: Dataset de entrenamiento optimizado.
            - val_ds: Dataset de validación optimizado.
            - class_names: Lista con los nombres de las clases.
    """
    print("--- Preparando Dataset ---")
    
    # Configuración de rutas
    KAGGLE_PATH = '/kaggle/input/videojuegos/images_dataset'
    LOCAL_PATH = './images_dataset' 

    if os.environ.get('KAGGLE_KERNEL_RUN_TYPE') is not None:
        print("Entorno detectado: Kaggle")
        data_dir = pathlib.Path(KAGGLE_PATH)
    else:
        print("Entorno detectado: Local")
        data_dir = pathlib.Path(LOCAL_PATH)
    
    # Carga inicial
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

    # División del dataset (Train 70% / Val 15% / Test 15%)
    n_batches = tf.data.experimental.cardinality(full_ds).numpy()
    train_size = int(0.7 * n_batches)
    val_size = int(0.15 * n_batches)

    train_ds = full_ds.take(train_size)
    remaining_ds = full_ds.skip(train_size)
    val_ds = remaining_ds.take(val_size)
    
    # Optimización (Cache & Prefetch)
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names

def build_transfer_model():
    """
    Construye la arquitectura del modelo basada en EfficientNetV2B0.
    Aplica Data Augmentation y congela la base pre-entrenada.

    Returns:
        tuple: (model, base_model)
            - model: El modelo Keras completo compilado.
            - base_model: La capa base de EfficientNet (para descongelar luego).
    """
    inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    
    # Data Augmentation
    x = layers.RandomFlip("horizontal_and_vertical")(inputs)
    x = layers.RandomRotation(0.2)(x)
    x = layers.RandomZoom(0.1)(x)
    
    # Modelo Base (Pre-entrenado)
    base_model = EfficientNetV2B0(
        include_top=False, 
        weights='imagenet', 
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
    )
    
    base_model.trainable = False # Congelado inicial
    
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x) 
    x = layers.Dropout(0.2)(x) 
    outputs = layers.Dense(10, activation='softmax')(x) 
    
    model = keras.Model(inputs, outputs, name="Transfer_EfficientNetV2")
    return model, base_model

def run_feature_extraction(model, train_ds, val_ds):
    """
    Fase 1: Feature Extraction.
    Entrena solo las capas superiores (clasificador) manteniendo la base congelada.

    Args:
        model: Modelo Keras construido.
        train_ds: Dataset de entrenamiento.
        val_ds: Dataset de validación.

    Returns:
        History: Objeto history del entrenamiento.
    """
    print("\n--- Fase 1: Feature Extraction ---")
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks_transfer = [
        callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        callbacks.ModelCheckpoint('best_transfer.keras', monitor='val_accuracy', save_best_only=True, verbose=0)
    ]

    history = model.fit(
        train_ds,
        epochs=20,
        validation_data=val_ds,
        callbacks=callbacks_transfer
    )
    return history

def run_fine_tuning(model, base_model, history_phase1, train_ds, val_ds):
    """
    Fase 2: Fine Tuning.
    Descongela el modelo base y re-entrena con un Learning Rate muy bajo.

    Args:
        model: Modelo Keras actual.
        base_model: Referencia al modelo base dentro del modelo principal.
        history_phase1: Historial de la fase 1 (para continuar épocas).
        train_ds: Dataset de entrenamiento.
        val_ds: Dataset de validación.

    Returns:
        History: Objeto history del fine tuning.
    """
    print("\n--- Fase 2: Fine Tuning ---")
    
    base_model.trainable = True

    # LR muy bajo para no romper pesos (1e-5)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks_finetune = [
        callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        callbacks.ModelCheckpoint('best_transfer.keras', monitor='val_accuracy', save_best_only=True, verbose=0)
    ]

    history = model.fit(
        train_ds,
        epochs=20, 
        initial_epoch=history_phase1.epoch[-1],
        validation_data=val_ds,
        callbacks=callbacks_finetune
    )
    return history

def plot_combined_history(history_tl, history_finetune):
    """
    Combina y grafica los historiales de entrenamiento de ambas fases.
    Muestra una línea vertical donde comenzó el Fine Tuning.
    """
    acc = history_tl.history['accuracy'] + history_finetune.history['accuracy']
    val_acc = history_tl.history['val_accuracy'] + history_finetune.history['val_accuracy']
    loss = history_tl.history['loss'] + history_finetune.history['loss']
    val_loss = history_tl.history['val_loss'] + history_finetune.history['val_loss']

    plt.figure(figsize=(12, 5))
    
    # Gráfica de Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    # Línea divisoria
    plt.plot([len(history_tl.history['accuracy'])-1, len(history_tl.history['accuracy'])-1], 
            plt.ylim(), label='Inicio Fine Tuning', ls='--', color='green') 
    plt.legend(loc='lower right')
    plt.title('Transfer Learning: Evolución de Precisión')
    plt.grid(True)

    # Gráfica de Loss
    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.plot([len(history_tl.history['loss'])-1, len(history_tl.history['loss'])-1], 
            plt.ylim(), label='Inicio Fine Tuning', ls='--', color='green')
    plt.legend(loc='upper right')
    plt.title('Transfer Learning: Evolución de Pérdida')
    plt.grid(True)
    
    plt.show()
    return val_acc

def print_final_report(model, training_time, final_acc):
    """
    Imprime un resumen final exclusivo del modelo entrenado.
    """
    params = model.count_params()
    
    print("\n" + "="*60)
    print("      REPORTE FINAL: TRANSFER LEARNING (EfficientNetV2)")
    print("="*60)
    print(f"Arquitectura Base: EfficientNetV2B0")
    print(f"Parámetros Totales: {params:,}")
    print(f"Tiempo Total:       {training_time:.2f} segundos")
    print(f"Mejor Val Accuracy: {final_acc:.4f}")
    print("="*60)

def main_pipeline():
    """
    Función orquestadora que ejecuta todo el flujo de trabajo.
    """
    # 1. Preparar Datos
    train_ds, val_ds, _ = prepare_dataset()
    
    # 2. Construir Modelo
    transfer_model, base_model = build_transfer_model()
    transfer_model.summary()
    
    start_time = time.time()
    
    # 3. Entrenamiento Fase 1 (Feature Extraction)
    history_tl = run_feature_extraction(transfer_model, train_ds, val_ds)
    
    # 4. Entrenamiento Fase 2 (Fine Tuning)
    history_finetune = run_fine_tuning(transfer_model, base_model, history_tl, train_ds, val_ds)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # 5. Resultados y Visualización
    full_val_acc = plot_combined_history(history_tl, history_finetune)
    best_acc = max(full_val_acc)
    
    print_final_report(transfer_model, total_time, best_acc)

if __name__ == "__main__":
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    
    # Ejecutar pipeline
    main_pipeline()