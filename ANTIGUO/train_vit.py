import train_utils as utils
import os

# Configuración de entorno
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from transformers import TFViTModel

# Configuración
utils.set_reproducibility()
# ViT consume mucha VRAM, forzamos batch bajo si es necesario
BATCH_SIZE_VIT = 4 

class ViTWrapper(layers.Layer):
    """Wrapper para modelo HuggingFace compatible con Keras."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.vit = TFViTModel.from_pretrained('google/vit-base-patch16-224', from_pt=True)
        self.vit.trainable = False 
        
    def call(self, inputs):
        return self.vit(pixel_values=inputs).pooler_output

def build_vit_classifier():
    inputs = keras.Input(shape=(utils.IMG_HEIGHT, utils.IMG_WIDTH, 3))
    
    # Data Augmentation (Sequential para evitar error de grafos)
    augmentation = keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.1),
    ], name="augmentation_block")
    x = augmentation(inputs)
    
    # Preprocesamiento Obligatorio para ViT
    x = layers.Resizing(224, 224)(x)          # ViT necesita 224x224
    x = layers.Rescaling(1./127.5, offset=-1)(x) # Pixeles entre -1 y 1
    x = layers.Permute((3, 1, 2))(x)          # De (H,W,C) a (C,H,W)
    
    x = ViTWrapper()(x)
    
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    
    return keras.Model(inputs, outputs, name="ViT_Classifier")

def run_vit_pipeline():
    # 1. Datos
    train_ds, val_ds, _ = utils.prepare_dataset(BATCH_SIZE_VIT)
    
    # 2. Construcción
    model = build_vit_classifier()
    model.summary()
    
    # 3. Entrenamiento
    print("\n--- Iniciando Entrenamiento ViT ---")
    start_time = time.time()
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    history = model.fit(
        train_ds, epochs=10, validation_data=val_ds,
        callbacks=[
            callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
            callbacks.ModelCheckpoint('best_vit.keras', save_best_only=True)
        ]
    )
    
    total_time = time.time() - start_time
    
    # 4. Reporte
    utils.plot_history(history, "Vision Transformer (ViT)")
    utils.print_final_report(model, total_time, max(history.history['val_accuracy']))

if __name__ == "__main__":
    tf.keras.backend.clear_session()
    run_vit_pipeline()