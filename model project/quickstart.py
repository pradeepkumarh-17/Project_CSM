#!/usr/bin/env python
"""
Quick Start Script for Emotion & Stress Predictor
Run this to choose and launch either interface
"""

import os
import sys
import subprocess
from pathlib import Path

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_banner():
    print("\n" + "="*60)
    print("🎭 EMOTION & STRESS LEVEL PREDICTOR")
    print("="*60)
    print("\nAvailable Interfaces:")
    print("  1) 🖥️  Desktop GUI (Tkinter)")
    print("     - Upload images or capture from webcam")
    print("     - Works offline, no browser needed")
    print("\n  2) 🌐 Web Interface (Flask)")
    print("     - Browser-based, mobile responsive")
    print("     - Drag-and-drop support")
    print("\n  3) 📖 View Documentation")
    print("     - Setup guide and tips")
    print("\n  4) ✅ Verify Installation")
    print("     - Check all dependencies")
    print("\n  5) ❌ Exit")
    print("\n" + "="*60)

def run_gui():
    print("\n▶️  Starting Desktop GUI...")
    print("   (This may take a few moments to load TensorFlow)\n")
    try:
        subprocess.run(
            [sys.executable, "emotion_predictor_gui.py"],
            cwd=os.getcwd()
        )
    except KeyboardInterrupt:
        print("\n⏸️  GUI closed by user")
    except Exception as e:
        print(f"\n❌ Error running GUI: {e}")

def run_web():
    print("\n▶️  Starting Web Server...")
    print("   (This may take a few moments to load TensorFlow)")
    print("\n📍 Visit: http://localhost:5000")
    print("   Press Ctrl+C to stop the server\n")
    
    # Check if Flask is installed
    try:
        import flask
    except ImportError:
        print("⚠️  Flask not installed. Installing now...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "flask", "pillow"],
            capture_output=True
        )
    
    try:
        subprocess.run(
            [sys.executable, "app.py"],
            cwd=os.getcwd()
        )
    except KeyboardInterrupt:
        print("\n⏸️  Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error running web interface: {e}")

def verify_installation():
    print("\n" + "="*60)
    print("VERIFYING INSTALLATION")
    print("="*60)
    
    # Check model file
    model_path = r"ml models\fer4_cnn.h5"
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"\n✅ Model: {model_path} ({size_mb:.1f} MB)")
    else:
        print(f"\n❌ Model not found: {model_path}")
        print("   Run face.py to train the model first")
    
    # Check interface files
    print("\n📁 Interface Files:")
    for file in ["emotion_predictor_gui.py", "app.py", "templates/index.html"]:
        if os.path.exists(file):
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file}")
    
    # Check dependencies
    print("\n📦 Dependencies:")
    deps = {
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'cv2': 'OpenCV',
        'tensorflow': 'TensorFlow',
        'PIL': 'Pillow',
        'flask': 'Flask'
    }
    
    for import_name, display_name in deps.items():
        try:
            __import__(import_name)
            print(f"   ✅ {display_name}")
        except ImportError:
            print(f"   ❌ {display_name} (not installed)")
    
    # Try to load model
    print("\n🔧 Testing Model Loading:")
    try:
        import tensorflow as tf
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            print(f"   ✅ Model loaded successfully")
            print(f"   ✅ Input shape: {model.input_shape}")
            print(f"   ✅ Parameters: {model.count_params():,}")
        else:
            print(f"   ⚠️  Model file not found")
    except Exception as e:
        print(f"   ❌ Could not load model: {e}")
    
    print("\n" + "="*60 + "\n")

def show_docs():
    doc_files = [
        ("SETUP_GUIDE.md", "Setup Guide & Features"),
        ("INTERFACE_README.md", "Detailed Documentation")
    ]
    
    print("\n" + "="*60)
    print("DOCUMENTATION")
    print("="*60)
    
    for file, desc in doc_files:
        if os.path.exists(file):
            print(f"\n✅ {file}")
            print(f"   {desc}")
            print(f"\n   Open with: notepad {file}")
        else:
            print(f"\n❌ {file} not found")
    
    print("\n" + "="*60 + "\n")

def main():
    while True:
        clear_screen()
        print_banner()
        
        choice = input("Select an option (1-5): ").strip()
        
        if choice == "1":
            clear_screen()
            run_gui()
        elif choice == "2":
            clear_screen()
            run_web()
        elif choice == "3":
            clear_screen()
            show_docs()
            input("Press Enter to return to menu...")
        elif choice == "4":
            clear_screen()
            verify_installation()
            input("Press Enter to return to menu...")
        elif choice == "5":
            print("\n👋 Goodbye!\n")
            sys.exit(0)
        else:
            print("\n❌ Invalid choice. Please try again.")
            input("Press Enter to continue...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!\n")
        sys.exit(0)
