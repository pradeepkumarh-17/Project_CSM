#!/usr/bin/env python
"""
Quick test script for Emotion Predictor Interface
Tests both GUI and web functionality
"""

import os
import sys
from pathlib import Path

print("=" * 60)
print("🎭 Emotion Predictor - Interface Test")
print("=" * 60)

# Check Python version
print(f"\n✓ Python Version: {sys.version}")

# Check required modules
modules_to_check = [
    'numpy', 'pandas', 'matplotlib', 'seaborn',
    'sklearn', 'cv2', 'tensorflow', 'flask', 'PIL'
]

print("\n📦 Checking Dependencies:")
missing_modules = []
for module_name in modules_to_check:
    try:
        __import__(module_name)
        print(f"  ✓ {module_name}")
    except ImportError:
        print(f"  ✗ {module_name} (MISSING)")
        missing_modules.append(module_name)

if missing_modules:
    print(f"\n⚠️  Missing modules: {', '.join(missing_modules)}")
    print("\nInstall with:")
    print(f"  pip install {' '.join(missing_modules)}")
else:
    print("\n✅ All dependencies installed!")

# Check model file
print("\n📁 Checking Files:")
model_path = r"ml models\fer4_cnn.h5"
if os.path.exists(model_path):
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"  ✓ Model file found: {model_path} ({size_mb:.1f} MB)")
else:
    print(f"  ✗ Model file not found: {model_path}")
    print("    Please run face.py to train the model first")

# Check interface files
interface_files = [
    'emotion_predictor_gui.py',
    'app.py',
    'templates/index.html',
    'INTERFACE_README.md'
]

for file in interface_files:
    if os.path.exists(file):
        print(f"  ✓ {file}")
    else:
        print(f"  ✗ {file} (MISSING)")

# Test model loading
print("\n🔧 Testing Model Loading:")
try:
    import tensorflow as tf
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        print(f"  ✓ Model loaded successfully")
        print(f"  ✓ Model input shape: {model.input_shape}")
        print(f"  ✓ Model output shape: {model.output_shape}")
        print(f"  ✓ Total parameters: {model.count_params():,}")
    else:
        print(f"  ✗ Cannot load model - file not found")
except Exception as e:
    print(f"  ✗ Error loading model: {e}")

print("\n" + "=" * 60)
print("🚀 Ready to Use!")
print("=" * 60)
print("\nTo start the Desktop GUI:")
print("  python emotion_predictor_gui.py")

print("\nTo start the Web Interface:")
print("  python app.py")
print("  Then visit: http://localhost:5000")

print("\nFor more information, see: INTERFACE_README.md")
print("=" * 60)
