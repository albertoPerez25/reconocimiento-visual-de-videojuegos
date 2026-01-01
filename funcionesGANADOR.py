# Predicción In The Wild
def predict_top_k(model, img_path, class_names, k=3):
    # Cargar y preprocesar la imagen
    img = tf.keras.utils.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Crear batch de 1

    # Predicción
    preds = model.predict(img_array, verbose=0)[0]
    
    # Obtener los índices de los top-k resultados
    top_k_indices = np.argsort(preds)[-k:][::-1]
    
    # Visualización
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title("Imagen de Prueba")

    plt.subplot(1, 2, 2)
    labels = [class_names[i] for i in top_k_indices]
    values = [preds[i] for i in top_k_indices]
    sns.barplot(x=values, y=labels, palette='viridis')
    plt.title(f"Top {k} Predicciones")
    plt.xlim(0, 1)
    plt.show()



# Test de robusted de ruido y desenfoque
import cv2

def test_robustness(model, img_array, class_names):
    # 1. Imagen Original
    # 2. Añadir Ruido Gaussiano
    noise = np.random.normal(0, 25, img_array.shape)
    noisy_img = np.clip(img_array + noise, 0, 255).astype('uint8')
    
    # 3. Añadir Desenfoque (Blur)
    blurred_img = cv2.GaussianBlur(img_array, (15, 15), 0)

    images = [img_array, noisy_img, blurred_img]
    titles = ["Original", "Con Ruido", "Desenfoque"]
    
    plt.figure(figsize=(15, 5))
    for i, img in enumerate(images):
        plt.subplot(1, 3, i+1)
        plt.imshow(img.astype('uint8'))
        
        # Predicción
        p_img = tf.expand_dims(img, 0)
        pred = model.predict(p_img, verbose=0)
        class_idx = np.argmax(pred)
        conf = np.max(pred)
        
        plt.title(f"{titles[i]}\nPred: {class_names[class_idx]} ({conf:.1%})")
        plt.axis('off')
    plt.show()



# Test de oclusión
def test_occlusion(model, img_array, class_names):
    h, w, _ = img_array.shape
    occluded_img = img_array.copy()
    
    # Tapar las esquinas (donde suele estar el HUD/UI)
    box_size = h // 4
    occluded_img[0:box_size, 0:box_size, :] = 0      # Top-left
    occluded_img[h-box_size:h, w-box_size:w, :] = 0  # Bottom-right
    
    # Predicción
    pred_orig = model.predict(tf.expand_dims(img_array, 0), verbose=0)
    pred_occl = model.predict(tf.expand_dims(occluded_img, 0), verbose=0)

    # Visualización
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(img_array.astype('uint8'))
    axes[0].set_title(f"Original: {class_names[np.argmax(pred_orig)]}")
    
    axes[1].imshow(occluded_img.astype('uint8'))
    axes[1].set_title(f"Ocluida: {class_names[np.argmax(pred_occl)]}")
    
    for ax in axes: ax.axis('off')
    plt.show()



# Visualización de capas intermedias
def plot_feature_maps(model, img_array, layer_index=2):
    # Crear un modelo que extraiga la salida de una capa específica
    feature_extractor = tf.keras.Model(inputs=model.inputs, outputs=model.layers[layer_index].output)
    
    # Obtener el mapa de características
    features = feature_extractor.predict(tf.expand_dims(img_array, 0), verbose=0)
    
    # Visualizar los primeros 16 filtros de esa capa
    plt.figure(figsize=(12, 12))
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.imshow(features[0, :, :, i], cmap='viridis')
        plt.axis('off')
    plt.suptitle(f"Activaciones de la Capa: {model.layers[layer_index].name}")
    plt.show()