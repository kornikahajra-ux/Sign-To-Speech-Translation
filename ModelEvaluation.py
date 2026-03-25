from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

def evaluate_and_plot_clean(model_path, X_val, y_val, label_map):
    
    model = tf.keras.models.load_model(model_path)
    
    
    predictions = model.predict(X_val, batch_size=4)
    y_pred = np.argmax(predictions, axis=1)

    
    cm = confusion_matrix(y_val, y_pred)
    
    
    classes = [name for name, idx in sorted(label_map.items(), key=lambda item: item[1])]

    
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title='Confusion Matrix: Dynamic Sign Recognition',
           ylabel='True Label',
           xlabel='Predicted Label')

    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")


    
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    plt.show()

    
    print("\nDetailed Performance Evaluation:\n")
    print(classification_report(y_val, y_pred, target_names=classes))


evaluate_and_plot_clean('Sign_Translation_Model_6.h5', X_val, y_val, label_map)