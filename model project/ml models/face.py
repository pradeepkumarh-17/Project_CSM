import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
import itertools
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, utils, preprocessing

DATA_CSV = r"D:\model project\dataset\fer2013 (1).csv"  # put fer2013.csv in same folder
IMG_SIZE = 48  # FER2013 original size
BATCH_SIZE = 64
EPOCHS = 30
RANDOM_STATE = 42
MODEL_OUT = r"ml models/fer4_cnn.h5"
LABEL_MAP_JSON = "label_map.json"

keep_labels = {
    3: "happy",
    6: "neutral",
    4: "sad",
    2: "fear"
}

stress_map = {
    "happy": "low",
    "neutral": "mid",
    "sad": "high",
    "fear": "high"
}

print("Loading CSV:", DATA_CSV)
df = pd.read_csv(DATA_CSV)

# Inspect basic structure
print("CSV columns:", df.columns.tolist())
# FER2013 CSV typically has 'emotion','pixels','Usage'

# Filter to selected labels
df = df[df['emotion'].isin(keep_labels.keys())].copy()
df['emotion_name'] = df['emotion'].map(keep_labels)

# Quick class counts
counts = df['emotion_name'].value_counts()
print("Class counts:\n", counts)

# Plot class distribution (EDA)
plt.figure(figsize=(6,4))
sns.barplot(x=counts.index, y=counts.values)
plt.title("Class distribution (4 emotions)")
plt.xlabel("Emotion")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("class_distribution.png")
plt.close()

def pixels_to_array(pix_string):
    # pixels are space-separated grayscale values
    arr = np.array(pix_string.split(), dtype=int)
    arr = arr.reshape(IMG_SIZE, IMG_SIZE)
    return arr

print("Converting pixels to arrays (this may take a few seconds)...")
images = np.stack(df['pixels'].apply(pixels_to_array).values)
labels = df['emotion_name'].values

# normalize to [0,1]
images = images.astype('float32') / 255.0
# expand channel dimension for Keras (H,W,1)
images = np.expand_dims(images, -1)

# Encode labels to integers
label_names = sorted(df['emotion_name'].unique().tolist())  # alphabetical order
label_to_idx = {name: idx for idx, name in enumerate(label_names)}
idx_to_label = {v:k for k,v in label_to_idx.items()}
labels_idx = np.array([label_to_idx[l] for l in labels])

# Train-test split (use stratify)
X_train, X_test, y_train, y_test = train_test_split(
    images, labels_idx, test_size=0.20, random_state=RANDOM_STATE, stratify=labels_idx
)
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

# Compute class weights to handle imbalance
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = {i: w for i, w in enumerate(class_weights)}
print("Class weights:", class_weights)

# Data augmentation generator
train_datagen = preprocessing.image.ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)
# For test - only rescale already done
test_datagen = preprocessing.image.ImageDataGenerator()

train_generator = train_datagen.flow(X_train, utils.to_categorical(y_train, num_classes=len(label_names)),
                                     batch_size=BATCH_SIZE, shuffle=True)
test_generator = test_datagen.flow(X_test, utils.to_categorical(y_test, num_classes=len(label_names)),
                                   batch_size=BATCH_SIZE, shuffle=False)

def build_cnn(input_shape=(IMG_SIZE, IMG_SIZE,1), n_classes=4):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(256, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(n_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model

model = build_cnn()
model.summary()

# Compile with Adam, learning rate schedule
opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks
es = callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1)
rlr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
mc = callbacks.ModelCheckpoint(MODEL_OUT, monitor='val_accuracy', save_best_only=True, verbose=1)

steps_per_epoch = len(X_train) // BATCH_SIZE
val_steps = len(X_test) // BATCH_SIZE

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,
    validation_data=test_generator,
    validation_steps=val_steps,
    class_weight=class_weights,
    callbacks=[es, rlr, mc]
)

y_pred_probs = model.predict(X_test, batch_size=BATCH_SIZE, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:\n", cm)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_names, yticklabels=label_names, cmap='Blues')
plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

# Classification report
report = classification_report(y_test, y_pred, target_names=label_names, digits=4)
print("Classification report:\n", report)
with open("classification_report.txt", "w") as f:
    f.write(report)
    
import matplotlib.pyplot as plt
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend(); plt.title('Loss')
plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.legend(); plt.title('Accuracy')
plt.tight_layout()
plt.show() # Display the plot directly

