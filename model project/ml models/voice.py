import os
import numpy as np
import pandas as pd
import librosa, librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import zipfile
import gradio as gr
import warnings
warnings.filterwarnings('ignore')

RAVDESS_PATH = r"dataset\audio_speech_actors_01-24"   # <-- Change this to your dataset path
SAMPLE_RATE = 22050
DURATION = 3

# Map emotion codes from RAVDESS format filename
# RAVDESS format: MM-VV-EE-II-SS-RR-AA.wav
# Position 3 (EE) = Emotion code
TARGET_EMOTIONS = {'01': 'neutral', '03': 'happy', '04': 'sad', '05': 'angry'}
STRESS_MAP = {'neutral': 'Medium', 'happy': 'Low', 'sad': 'High', 'angry': 'High'}

all_mfccs = []
all_labels = []

# Define the target length for MFCCs based on SAMPLE_RATE and DURATION
# librosa.feature.mfcc by default uses hop_length=512
target_mfcc_length = 1 + int((SAMPLE_RATE * DURATION) / 512)

print(f"Expected MFCC feature shape (n_mfcc, n_frames): (40, {target_mfcc_length})")
print(f"Loading audio files from: {RAVDESS_PATH}")

# Handle both flat directory and subdirectory structures
audio_files = []
if os.path.isdir(RAVDESS_PATH):
    # Check subdirectories
    for actor_dir in sorted(os.listdir(RAVDESS_PATH)):
        actor_path = os.path.join(RAVDESS_PATH, actor_dir)
        if os.path.isdir(actor_path):
            for filename in sorted(os.listdir(actor_path)):
                if filename.endswith('.wav'):
                    audio_files.append(os.path.join(actor_path, filename))
    
    # If no files found in subdirectories, check root directory
    if not audio_files:
        for filename in os.listdir(RAVDESS_PATH):
            file_path = os.path.join(RAVDESS_PATH, filename)
            if os.path.isfile(file_path) and filename.endswith('.wav'):
                audio_files.append(file_path)

print(f"Found {len(audio_files)} audio files")

for file_path in sorted(audio_files):
    filename = os.path.basename(file_path)
    if filename.endswith('.wav'):
        # Extract emotion code from RAVDESS filename
        # RAVDESS format: MM-VV-EE-II-SS-RR-AA.wav
        # Emotion code is at position 2 (index 2) when split by '-'
        
        emotion_label = None
        parts = filename.replace('.wav', '').split('-')
        
        # Extract emotion code (position 3, index 2 in 0-based indexing)
        if len(parts) >= 3:
            emotion_code = parts[2]  # Get emotion code
            if emotion_code in TARGET_EMOTIONS:
                emotion_label = TARGET_EMOTIONS[emotion_code]
        
        if emotion_label:
            file_path_final = file_path

            try:
                # Load audio file
                y, sr = librosa.load(file_path_final, sr=SAMPLE_RATE, duration=DURATION)

                # Extract MFCCs
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

                # Ensure consistent length by padding or truncating
                if mfccs.shape[1] < target_mfcc_length:
                    # Pad if shorter
                    pad_width = target_mfcc_length - mfccs.shape[1]
                    mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
                elif mfccs.shape[1] > target_mfcc_length:
                    # Truncate if longer
                    mfccs = mfccs[:, :target_mfcc_length]

                all_mfccs.append(mfccs)
                all_labels.append(emotion_label)

            except Exception as e:
                print(f"Error processing {file_path_final}: {e}")

print(f"Successfully processed {len(all_mfccs)} audio files.")
print(f"Shape of one MFCC feature set: {all_mfccs[0].shape if all_mfccs else 'N/A'}")

df_emotions = pd.DataFrame({'Emotion': all_labels})

plt.figure(figsize=(8, 6))
sns.countplot(data=df_emotions, x='Emotion', palette='viridis')
plt.title('Distribution of Emotions in the Dataset')
plt.xlabel('Emotion')
plt.ylabel('Count')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

def add_white_noise(data, noise_amount=0.005):
    noise = np.random.randn(len(data)) * noise_amount
    data_noise = data + noise
    return data_noise

def pitch_shift(data, sr, n_steps=4):
    return librosa.effects.pitch_shift(y=data, sr=sr, n_steps=n_steps)

def time_stretch(data, rate=0.8):
    # Using librosa.effects.time_stretch requires data to be converted to float
    # and then back to its original type if necessary, but librosa functions usually handle float32.
    return librosa.effects.time_stretch(y=data, rate=rate)

augmented_mfccs = []
augmented_labels = []

# Apply augmentation to loaded audio data (simple noise augmentation)
print("Applying data augmentation...")
augmented_count = 0
for i, (mfcc, label) in enumerate(zip(all_mfccs, all_labels)):
    if i % 100 == 0:
        print(f"Augmenting sample {i}/{len(all_mfccs)}")
    
    # Add only one augmented version per original sample for speed
    augmented_mfcc = mfcc + np.random.normal(0, 0.002, mfcc.shape)
    augmented_mfccs.append(augmented_mfcc)
    augmented_labels.append(label)
    augmented_count += 1

print(f"Data augmentation completed. Added {augmented_count} augmented samples.")

all_mfccs.extend(augmented_mfccs)
all_labels.extend(augmented_labels)

print(f"Total original + augmented files processed: {len(all_mfccs)}")
print(f"Total original + augmented labels: {len(all_labels)}")

X = np.array(all_mfccs)
y = np.array(all_labels)

# Reshape X for StandardScaler: flatten each MFCC array into a 1D vector
X_reshaped = X.reshape(X.shape[0], -1)

# Initialize StandardScaler
scaler = StandardScaler()

# Fit and transform the features
X_scaled = scaler.fit_transform(X_reshaped)

print(f"Shape of MFCC features before scaling: {X.shape}")
print(f"Shape of MFCC features after reshaping for scaling: {X_reshaped.shape}")
print(f"Shape of MFCC features after scaling: {X_scaled.shape}")
print(f"Shape of labels: {y.shape}")

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_one_hot = to_categorical(y_encoded)

# Reshape X_scaled back to 3D for LSTM input
X_reshaped_for_lstm = X_scaled.reshape(X_scaled.shape[0], 40, target_mfcc_length)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_reshaped_for_lstm, y_one_hot, test_size=0.2, random_state=42)

print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of y_test: {y_test.shape}")
print(f"Number of classes: {y_one_hot.shape[1]}")

from tensorflow.keras.optimizers import Adam

# Get input shape and number of classes from the preprocessed data
input_shape = (X_train.shape[1], X_train.shape[2])
num_classes = y_train.shape[1]

# 1. Initialize a Sequential model
model = Sequential()

# 2. Add an LSTM layer with 128 units, returning sequences, and specifying the input_shape
model.add(LSTM(128, return_sequences=True, input_shape=input_shape))

# 3. Add a Dropout layer with a rate of 0.3
model.add(Dropout(0.3))

# 4. Add another LSTM layer with 64 units
model.add(LSTM(64))

# 5. Add a Dense layer with 32 units and a ReLU activation function
model.add(Dense(32, activation='relu'))

# 6. Add a final Dense output layer with num_classes units and a softmax activation function
model.add(Dense(num_classes, activation='softmax'))

# 7. Compile the model
optimizer = Adam(learning_rate=1e-3) # Learning rate 0.001
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

import keras_tuner as kt

# Skip Hyperband tuning due to weight loading issues
# Instead, train a single optimized model directly

print("Building optimized LSTM model...")

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Create an optimized model without hyperparameter tuning
best_model = Sequential()
best_model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
best_model.add(Dropout(0.2))
best_model.add(LSTM(64))
best_model.add(Dense(32, activation='relu'))
best_model.add(Dense(num_classes, activation='softmax'))

optimizer = Adam(learning_rate=1e-3)
best_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
print("Model compiled successfully.")

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

history = best_model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test), callbacks=[early_stopping, reduce_lr])
print("Model training completed.")

plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()
print("Training and validation accuracy plot generated.")

# Generate predictions for evaluation
y_test_numerical = np.argmax(y_test, axis=1)
y_pred = np.argmax(best_model.predict(X_test), axis=1)

conf_matrix = confusion_matrix(y_test_numerical, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.xlabel('Predicted Emotion')
plt.ylabel('True Emotion')
plt.title('Confusion Matrix')
plt.show()
print("Confusion matrix generated.")

class_report = classification_report(y_test_numerical, y_pred, target_names=encoder.classes_)
print("Classification Report:")
print(class_report)

best_model.save('ml models/cognitive_lstm_final_87.h5')
print("Model saved to 'ml models/cognitive_lstm_final_87.h5'.")