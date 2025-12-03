import train_utils as utils

# Configuración de entorno
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.applications import EfficientNetV2B0

# Configuración
utils.set_reproducibility()
BATCH_SIZE = 16

def build_transfer_model():
    """Construye EfficientNetV2B0 con Data Augmentation encapsulado."""
    inputs = keras.Input(shape=(utils.IMG_HEIGHT, utils.IMG_WIDTH, 3))
    
    # Data Augmentation en bloque Sequential (Fix para TF 2.16+)
    augmentation = keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.1),
    ], name="augmentation_block")
    
    x = augmentation(inputs)
    
    # Modelo Base
    base_model = EfficientNetV2B0(
        include_top=False, 
        weights='imagenet', 
        input_shape=(utils.IMG_HEIGHT, utils.IMG_WIDTH, 3)
    )
    base_model.trainable = False
    
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x) 
    x = layers.Dropout(0.2)(x) 
    outputs = layers.Dense(10, activation='softmax')(x) 
    
    return keras.Model(inputs, outputs, name="Transfer_EfficientNetV2"), base_model

def run_efficientnet_pipeline():
    # 1. Datos
    train_ds, val_ds, _ = utils.prepare_dataset(BATCH_SIZE)
    
    # 2. Construcción
    model, base_model = build_transfer_model()
    model.summary()
    
    start_time = time.time()
    
    # 3. Fase 1: Feature Extraction
    print("\n--- Fase 1: Feature Extraction ---")
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    history_tl = model.fit(
        train_ds, epochs=20, validation_data=val_ds,
        callbacks=[
            callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            callbacks.ModelCheckpoint('best_effnet.keras', save_best_only=True)
        ]
    )

    # 4. Fase 2: Fine Tuning
    print("\n--- Fase 2: Fine Tuning ---")
    base_model.trainable = True
    model.compile(optimizer=keras.optimizers.Adam(1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    history_finetune = model.fit(
        train_ds, epochs=20, initial_epoch=history_tl.epoch[-1], validation_data=val_ds,
        callbacks=[
            callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            callbacks.ModelCheckpoint('best_effnet.keras', save_best_only=True)
        ]
    )
    
    total_time = time.time() - start_time
    
    # 5. Reporte
    full_acc = utils.plot_combined_history(history_tl, history_finetune)
    utils.print_final_report(model, total_time, max(full_acc))

if __name__ == "__main__":
    tf.keras.backend.clear_session()
    run_efficientnet_pipeline()