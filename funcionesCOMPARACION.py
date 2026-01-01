# Tabla datos
def build_comparison_dataframe(models_dict, histories_dict, execution_times):
    """
    models_dict: {'CNN': cnn_model, 'ViT': vit_model, ...}
    histories_dict: {'CNN': history_cnn, 'ViT': history_vit, ...}
    execution_times: {'CNN': 120.5, 'ViT': 450.2, ...} (en segundos)
    """
    data = []
    
    for name in models_dict.keys():
        model = models_dict[name]
        hist = histories_dict[name].history if hasattr(histories_dict[name], 'history') else histories_dict[name]
        
        # Extraer métricas finales (última época)
        train_acc = hist['accuracy'][-1]
        val_acc = hist['val_accuracy'][-1]
        val_f1 = hist['val_f1_score'][-1]
        
        # Calcular complejidad
        params = model.count_params()
        size_mb = params * 4 / (1024**2) # Estimación en MB (float32)
        
        # Calcular ratio de overfitting
        overfitting = (train_acc - val_acc) / train_acc
        
        data.append({
            'Modelo': name,
            'Val Accuracy': val_acc,
            'Val F1-Score': val_f1,
            'Parámetros': params,
            'Tamaño (MB)': size_mb,
            'Tiempo (s)': execution_times.get(name, 0),
            'Overfitting (%)': overfitting * 100
        })
    
    return pd.DataFrame(data).sort_values(by='Val Accuracy', ascending=False)

# Ejemplo de uso:
# df_comparativo = build_comparison_dataframe(models, results, times)
# display(df_comparativo)



# Gráfica de eficiencia 
def plot_efficiency_comparison(df):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Parámetros', y='Val Accuracy', hue='Modelo', s=200)
    
    plt.title('Eficiencia del Modelo: Precisión vs. Complejidad')
    plt.xscale('log') # Escala logarítmica para ver mejor la diferencia de parámetros
    plt.grid(True, which="both", ls="-", alpha=0.5)
    
    # Añadir etiquetas a cada punto
    for i in range(df.shape[0]):
        plt.text(df.Parámetros.iloc[i], df['Val Accuracy'].iloc[i] + 0.01, 
                 df.Modelo.iloc[i], fontsize=12, ha='center')
    
    plt.show()



# Estabilidad y convergencia
def plot_multi_model_convergence(histories_dict):
    plt.figure(figsize=(12, 6))
    
    for name, hist_obj in histories_dict.items():
        hist = hist_obj.history if hasattr(hist_obj, 'history') else hist_obj
        plt.plot(hist['val_accuracy'], label=f'Val Acc: {name}', linewidth=2)
        
    plt.title('Comparativa de Convergencia (Validation Accuracy)')
    plt.xlabel('Épocas')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()



# Overffiting
def plot_overfitting_comparison(df):
    plt.figure(figsize=(10, 5))
    sns.barplot(data=df, x='Modelo', y='Overfitting (%)', palette='magma')
    
    plt.title('Análisis de Sobreajuste (Ratio de Overfitting)')
    plt.ylabel('Diferencia Train-Val (%)')
    plt.axhline(10, ls='--', color='red', label='Límite deseado (10%)') # Referencia común
    plt.legend()
    plt.show()



# Comparativa matrices de confusión
def compare_confusion_matrices(models_dict, test_ds, class_names):
    num_models = len(models_dict)
    fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 5))
    
    # Obtener etiquetas reales (solo una vez)
    y_true = np.concatenate([y for x, y in test_ds], axis=0)
    
    for i, (name, model) in enumerate(models_dict.items()):
        y_pred = np.argmax(model.predict(test_ds, verbose=0), axis=1)
        cm = confusion_matrix(y_true, y_pred, normalize='true')
        
        sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', ax=axes[i], cbar=False,
                    xticklabels=class_names, yticklabels=class_names)
        axes[i].set_title(f'Matriz: {name}')
        axes[i].set_xlabel('Predicción')
        if i == 0: axes[i].set_ylabel('Real')
        
    plt.tight_layout()
    plt.show()