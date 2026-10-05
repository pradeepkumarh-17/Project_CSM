## 🎭 Emotion & Stress Level Predictor

This project includes two interfaces for predicting emotions and stress levels from images using a trained CNN model.

### 📋 Setup Instructions

#### 1. Install Additional Dependencies

For the web interface (Flask):

```bash
pip install flask pillow
```

For the desktop GUI (already has most dependencies):

- Tkinter comes with Python by default

### 🖥️ Option 1: Desktop GUI Interface

**How to run:**

```bash
cd d:\model project
python emotion_predictor_gui.py
```

**Features:**

- 📁 Upload images from disk
- 📷 Capture images from webcam in real-time
- 🎨 Beautiful Tkinter interface
- 📊 Real-time emotion probabilities visualization
- 🎯 Confidence score with progress bars
- 💾 Works offline (after model is trained)

**Interface Layout:**

- **Left side**: Image preview, upload/camera buttons
- **Right side**:
  - Detected emotion (large, color-coded)
  - Confidence level (progress bar)
  - Stress level (Low/Medium/High)
  - Individual emotion probabilities for all 4 emotions

**Stress Level Mapping:**

- 😊 Happy → **Low Stress**
- 😐 Neutral → **Medium Stress**
- 😢 Sad → **High Stress**
- 😨 Fear → **High Stress**

### 🌐 Option 2: Web Interface (Flask)

**How to run:**

```bash
cd d:\model project
pip install flask pillow
python app.py
```

Then open your browser and go to: **http://localhost:5000**

**Features:**

- 🌐 Access from any browser
- 📱 Mobile responsive design
- 🎨 Modern gradient UI with animations
- 🔄 Real-time drag-and-drop upload
- 📊 Beautiful visualization of results
- ☁️ Can be deployed to cloud servers

**How to use:**

1. Click or drag-and-drop an image
2. Click "Predict Emotion" button
3. View results with:
   - Detected emotion with color coding
   - Confidence percentage with progress bar
   - Stress level classification
   - Individual probabilities for all emotions

### 📊 Emotion Categories

The model recognizes 4 emotions:

1. **Happy** 😊 - Low stress, positive mood
2. **Neutral** 😐 - Medium stress, calm/expressionless
3. **Sad** 😢 - High stress, negative mood
4. **Fear** 😨 - High stress, anxious/scared

### 🎯 Supported Image Formats

- ✅ JPG / JPEG
- ✅ PNG
- ✅ BMP
- ✅ GIF

### 📸 Camera Capture (Desktop Only)

The desktop GUI includes a "Capture from Camera" button:

1. Click the button
2. Allow camera access
3. A camera window will open
4. Press **SPACE** to capture
5. Press **ESC** to cancel

### 🔧 Model Information

- **Architecture**: Convolutional Neural Network (CNN)
- **Input Size**: 48x48 grayscale images
- **Total Parameters**: 3.5M
- **Model File**: `ml models/fer4_cnn.h5`
- **Training Dataset**: FER2013 (Facial Expression Recognition 2013)

### 💡 Tips for Best Results

1. **Face Detection**: Ensure the face is clearly visible
2. **Image Quality**: Use well-lit, clear images
3. **Face Position**: Center the face in the image
4. **File Size**: Images up to 16MB are supported
5. **Format**: Use common formats (JPG, PNG)

### 📝 File Structure

```
d:\model project\
├── ml models\
│   └── fer4_cnn.h5              (trained model)
├── dataset\
│   └── fer2013 (1).csv          (training data)
├── emotion_predictor_gui.py      (desktop GUI)
├── app.py                        (Flask web server)
├── templates\
│   └── index.html               (web interface)
├── face.py                       (training script)
└── README.md                     (this file)
```

### 🚀 Deployment

**To deploy the web version to a cloud server:**

1. Use Flask with Gunicorn:

   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 app.py
   ```

2. Or use Docker:
   ```dockerfile
   FROM python:3.10
   WORKDIR /app
   COPY . .
   RUN pip install -r requirements.txt
   EXPOSE 5000
   CMD ["python", "app.py"]
   ```

### ❌ Troubleshooting

**"Model not found" error:**

- Run `face.py` first to train the model
- Ensure the model path is correct in the config

**Camera not working:**

- Check camera permissions
- Ensure no other app is using the camera
- Try restarting the application

**Image won't upload (web):**

- Check file size (max 16MB)
- Verify file format is supported
- Check browser console for errors

**Slow predictions:**

- This is normal for the first prediction (TensorFlow initialization)
- Subsequent predictions will be faster

### 📞 Contact & Support

For questions or issues, check the model training logs in `face.py`

---

**Made with ❤️ using TensorFlow & Keras**
