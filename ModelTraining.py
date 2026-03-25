import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

def prepare_data_functional(data_path, target_frames=40, img_size=(128, 128)):
    sequences = []
    labels = []
    
     
    label_names = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])
    label_map = {name: i for i, name in enumerate(label_names)}

    for label_name in label_names:
        label_dir = os.path.join(data_path, label_name)
        for seq_folder in os.listdir(label_dir):
            seq_path = os.path.join(label_dir, seq_folder)
            frames = sorted([os.path.join(seq_path, f) for f in os.listdir(seq_path) if f.endswith('.jpg')])
            
            if not frames: continue

            
            if len(frames) >= target_frames:
                
                idx = np.linspace(0, len(frames) - 1, target_frames, dtype=int)
                selected = [frames[i] for i in idx]
            else:
                
                selected = list(frames)
                while len(selected) < target_frames:
                    selected.append(frames[-1])
            
            
            temp_seq = []
            for f in selected:
                img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, img_size)
                temp_seq.append(img.astype('float32') / 255.0)
            
            sequences.append(np.array(temp_seq))
            labels.append(label_map[label_name])

    
    X_train, X_val, y_train, y_val = train_test_split(
        np.array(sequences), np.array(labels), test_size=0.2, stratify=labels, random_state=42
    )
    
    
    X_train = np.expand_dims(X_train, axis=-1)
    X_val = np.expand_dims(X_val, axis=-1)
    
    return X_train, X_val, y_train, y_val, label_map




def build_cnn_feature_extractor(input_shape=(128, 128, 1)):
    inputs = tf.keras.Input(shape=input_shape)

    # Conv Block 1
    x = tf.keras.layers.Conv2D(16, (3,3), padding='same', activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)

    # Conv Block 2
    x = tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)

    # Conv Block 3 
    x = tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)

    
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)

    return tf.keras.Model(inputs, x)


def build_sign_language_model(num_classes, input_shape=(40, 128, 128, 1)):
    inputs = tf.keras.Input(shape=input_shape)

    cnn = build_cnn_feature_extractor((128, 128, 1))

    x = tf.keras.layers.TimeDistributed(cnn)(inputs)

    x = tf.keras.layers.GRU(64, return_sequences=False)(x)

    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model





datapath = r"C:\Users\Kornika\Desktop\Files\Society Tasks\MACS DTU\SignToSpeech\Training\PreprocessedMotionData"
X_train, X_val, y_train, y_val, label_map = prepare_data_functional(datapath)

model = build_sign_language_model(num_classes=len(label_map))
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0005,
        weight_decay=1e-4
    ),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5),
    tf.keras.callbacks.ModelCheckpoint('Sign_Translation_Model_6.h5', save_best_only=True)
]

model.fit(X_train, y_train, 
          validation_data=(X_val, y_val), 
          epochs=50, 
          batch_size=4, 
          callbacks=callbacks)