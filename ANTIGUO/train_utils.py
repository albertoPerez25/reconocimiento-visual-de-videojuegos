import os

# Configuración de entorno
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import pathlib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# --- CONSTANTES GLOBALES ---
IMG_HEIGHT = 480
IMG_WIDTH = 854
SEED = 2025

def set_reproducibility():
    """Configura semillas y variables de entorno para consistencia."""
    os.environ["TF_USE_LEGACY_KERAS"] = "1"
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")

def prepare_dataset(batch_size):
    """
    Carga y optimiza el dataset detectando el entorno (Kaggle/Local).
    Acepta batch_size dinámico porque ViT consume más memoria que EfficientNet.
    """
    print(f"--- Preparando Dataset (Batch Size: {batch_size}) ---")
    
    KAGGLE_PATH = '/kaggle/input/eurosat-dataset/EuroSAT'
    LOCAL_PATH = './images_dataset' 

    if os.environ.get('KAGGLE_KERNEL_RUN_TYPE') is not None:
        data_dir = pathlib.Path(KAGGLE_PATH)
        print("Entorno: Kaggle")
    else:
        data_dir = pathlib.Path(LOCAL_PATH)
        print("Entorno: Local")
    
    # Carga con validación split (70% Train, 15% Val, 15% Test implícito)
    # Usamos subset="training" y "validation" para simplificar
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.15,
        subset="training",
        seed=SEED,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=batch_size,
        label_mode='int'
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.15,
        subset="validation",
        seed=SEED,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=batch_size,
        label_mode='int'
    )

    class_names = train_ds.class_names
    
    # Optimización (Cache & Prefetch)
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names

def print_final_report(model, training_time, final_acc):
    """Imprime un resumen estandarizado."""
    params = model.count_params()
    print("\n" + "="*60)
    print(f"      REPORTE FINAL: {model.name}")
    print("="*60)
    print(f"Parámetros Totales: {params:,}")
    print(f"Tiempo Total:       {training_time:.2f} segundos")
    print(f"Mejor Val Accuracy: {final_acc:.4f}")
    print("="*60)

# --- GRÁFICAS ---

def plot_history(history, title="Model History"):
    """Gráfica simple para entrenamientos de una sola fase (como ViT)."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.title(f'{title} - Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.title(f'{title} - Loss')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

def plot_combined_history(history_tl, history_finetune):
    """Gráfica compleja para Transfer Learning (Extraction + Fine Tuning)."""
    acc = history_tl.history['accuracy'] + history_finetune.history['accuracy']
    val_acc = history_tl.history['val_accuracy'] + history_finetune.history['val_accuracy']
    loss = history_tl.history['loss'] + history_finetune.history['loss']
    val_loss = history_tl.history['val_loss'] + history_finetune.history['val_loss']

    plt.figure(figsize=(12, 5))
    
    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.plot([len(history_tl.history['accuracy'])-1, len(history_tl.history['accuracy'])-1], 
            plt.ylim(), label='Inicio Fine Tuning', ls='--', color='green') 
    plt.legend(loc='lower right')
    plt.title('Evolución de Precisión (TL + FineTuning)')
    plt.grid(True)

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.plot([len(history_tl.history['loss'])-1, len(history_tl.history['loss'])-1], 
            plt.ylim(), label='Inicio Fine Tuning', ls='--', color='green')
    plt.legend(loc='upper right')
    plt.title('Evolución de Pérdida (TL + FineTuning)')
    plt.grid(True)
    
    plt.show()
    return val_acc