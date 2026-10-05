import os
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Check dataset path
RAVDESS_PATH = r"dataset\audio_speech_actors_01-24"
print(f"Checking path: {RAVDESS_PATH}")
print(f"Path exists: {os.path.isdir(RAVDESS_PATH)}")

if os.path.isdir(RAVDESS_PATH):
    # List subdirectories
    subdirs = [d for d in os.listdir(RAVDESS_PATH) if os.path.isdir(os.path.join(RAVDESS_PATH, d))]
    print(f"\nFound {len(subdirs)} subdirectories:")
    for d in sorted(subdirs)[:5]:  # Show first 5
        print(f"  - {d}")
    
    # Count audio files
    total_files = 0
    for actor_dir in os.listdir(RAVDESS_PATH):
        actor_path = os.path.join(RAVDESS_PATH, actor_dir)
        if os.path.isdir(actor_path):
            files = [f for f in os.listdir(actor_path) if f.endswith('.wav')]
            total_files += len(files)
    
    print(f"\nTotal WAV files: {total_files}")
    
    # Test loading a few files
    print("\nTesting audio file loading:")
    loaded = 0
    errors = 0
    
    TARGET_EMOTIONS = {'01': 'neutral', '03': 'happy', '04': 'sad', '05': 'angry', 
                       '02': 'calm', '06': 'fearful', '07': 'disgust', '08': 'surprised'}
    
    all_labels = []
    
    for actor_dir in sorted(os.listdir(RAVDESS_PATH))[:3]:  # Just first 3 actors
        actor_path = os.path.join(RAVDESS_PATH, actor_dir)
        if os.path.isdir(actor_path):
            for filename in sorted(os.listdir(actor_path))[:3]:  # First 3 files
                if filename.endswith('.wav'):
                    file_path = os.path.join(actor_path, filename)
                    try:
                        # Extract emotion code
                        parts = filename.replace('.wav', '').split('-')
                        if len(parts) > 1 and parts[1].isdigit():
                            emotion_code = parts[1]
                            if emotion_code in TARGET_EMOTIONS:
                                emotion_label = TARGET_EMOTIONS[emotion_code]
                                
                                # Try to load
                                y, sr = librosa.load(file_path, sr=22050, duration=3)
                                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
                                
                                loaded += 1
                                all_labels.append(emotion_label)
                                print(f"  ✓ {filename} -> {emotion_label} (shape: {mfcc.shape})")
                    except Exception as e:
                        errors += 1
                        print(f"  ✗ {filename}: {e}")
    
    print(f"\nLoaded: {loaded}, Errors: {errors}")
    print(f"Emotion distribution: {set(all_labels)}")

else:
    print("Dataset path not found!")
