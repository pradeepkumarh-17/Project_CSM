#!/usr/bin/env python
import sys
print("Python version:", sys.version)

try:
    import numpy as np
    print("✓ NumPy imported")
except Exception as e:
    print("✗ NumPy error:", e)

try:
    import pandas as pd
    print("✓ Pandas imported")
except Exception as e:
    print("✗ Pandas error:", e)

try:
    import matplotlib.pyplot as plt
    print("✓ Matplotlib imported")
except Exception as e:
    print("✗ Matplotlib error:", e)

try:
    import seaborn as sns
    print("✓ Seaborn imported")
except Exception as e:
    print("✗ Seaborn error:", e)

try:
    from sklearn.model_selection import train_test_split
    print("✓ Scikit-learn imported")
except Exception as e:
    print("✗ Scikit-learn error:", e)

try:
    import cv2
    print("✓ OpenCV imported")
except Exception as e:
    print("✗ OpenCV error:", e)

try:
    import tensorflow as tf
    print("✓ TensorFlow imported")
    print("  TensorFlow version:", tf.__version__)
except Exception as e:
    print("✗ TensorFlow error:", e)

# Test CSV loading
try:
    DATA_CSV = r"D:\model project\dataset\fer2013 (1).csv"
    print(f"\nTesting CSV load from: {DATA_CSV}")
    df = pd.read_csv(DATA_CSV)
    print(f"✓ CSV loaded successfully")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {df.columns.tolist()}")
except Exception as e:
    print(f"✗ CSV load error: {e}")
    import traceback
    traceback.print_exc()

print("\nAll imports and basic tests completed!")
