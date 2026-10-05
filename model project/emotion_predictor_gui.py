import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
import tensorflow as tf
from pathlib import Path
import os

class EmotionPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion & Stress Level Predictor")
        self.root.geometry("900x700")
        self.root.resizable(True, True)
        
        # Configure style
        self.root.config(bg="#f0f0f0")
        
        # Model path
        self.model_path = r"ml models\fer4_cnn.h5"
        self.IMG_SIZE = 48
        
        # Emotion to stress mapping
        self.stress_map = {
            "fear": "High",
            "sad": "High",
            "neutral": "Medium",
            "happy": "Low"
        }
        
        # Emotion colors for visualization
        self.emotion_colors = {
            "happy": "#4CAF50",    # Green
            "neutral": "#2196F3",   # Blue
            "sad": "#FF9800",       # Orange
            "fear": "#F44336"       # Red
        }
        
        # Load model
        self.model = None
        self.load_model()
        
        # Selected image path
        self.selected_image = None
        self.original_image = None
        
        # Create UI
        self.create_ui()
        
    def load_model(self):
        """Load the trained CNN model"""
        try:
            if os.path.exists(self.model_path):
                self.model = tf.keras.models.load_model(self.model_path)
                print(f"✓ Model loaded from {self.model_path}")
            else:
                messagebox.showerror("Error", f"Model not found at {self.model_path}\nPlease train the model first.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")
    
    def create_ui(self):
        """Create the user interface"""
        # Title
        title_frame = tk.Frame(self.root, bg="#2c3e50")
        title_frame.pack(fill=tk.X)
        
        title_label = tk.Label(title_frame, text="🎭 Emotion & Stress Level Predictor", 
                              font=("Helvetica", 18, "bold"), fg="white", bg="#2c3e50", pady=10)
        title_label.pack()
        
        # Main container
        main_frame = tk.Frame(self.root, bg="#f0f0f0")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left frame - Image display
        left_frame = tk.Frame(main_frame, bg="white")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        tk.Label(left_frame, text="Image Preview", font=("Helvetica", 12, "bold"), bg="white").pack(pady=5)
        
        # Image display canvas
        self.image_canvas = tk.Label(left_frame, bg="#e0e0e0", width=30, height=20)
        self.image_canvas.pack(padx=10, pady=10)
        
        # Button frame
        button_frame = tk.Frame(left_frame, bg="white")
        button_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.upload_btn = tk.Button(button_frame, text="📁 Upload Image", 
                                    command=self.upload_image,
                                    bg="#2196F3", fg="white", font=("Helvetica", 10, "bold"),
                                    padx=10, pady=8)
        self.upload_btn.pack(side=tk.LEFT, padx=5)
        
        self.camera_btn = tk.Button(button_frame, text="📷 Capture from Camera",
                                    command=self.capture_from_camera,
                                    bg="#FF9800", fg="white", font=("Helvetica", 10, "bold"),
                                    padx=10, pady=8)
        self.camera_btn.pack(side=tk.LEFT, padx=5)
        
        # Right frame - Results
        right_frame = tk.Frame(main_frame, bg="white")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        tk.Label(right_frame, text="Prediction Results", font=("Helvetica", 12, "bold"), bg="white").pack(pady=5)
        
        # Results display
        self.results_frame = tk.Frame(right_frame, bg="white")
        self.results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Emotion label and bar
        emotion_label_frame = tk.Frame(self.results_frame, bg="white")
        emotion_label_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(emotion_label_frame, text="Detected Emotion:", font=("Helvetica", 11, "bold"), bg="white").pack(anchor=tk.W)
        
        self.emotion_result = tk.Label(emotion_label_frame, text="--", 
                                      font=("Helvetica", 20, "bold"), bg="white", fg="#2196F3")
        self.emotion_result.pack(anchor=tk.W, pady=5)
        
        # Confidence bar
        confidence_frame = tk.Frame(self.results_frame, bg="white")
        confidence_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(confidence_frame, text="Confidence:", font=("Helvetica", 11, "bold"), bg="white").pack(anchor=tk.W)
        
        self.confidence_bar = tk.Canvas(confidence_frame, height=20, bg="#e0e0e0", highlightthickness=0)
        self.confidence_bar.pack(fill=tk.X, pady=5)
        
        self.confidence_text = tk.Label(confidence_frame, text="0%", font=("Helvetica", 10), bg="white")
        self.confidence_text.pack(anchor=tk.W)
        
        # Stress level
        stress_label_frame = tk.Frame(self.results_frame, bg="white")
        stress_label_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(stress_label_frame, text="Stress Level:", font=("Helvetica", 11, "bold"), bg="white").pack(anchor=tk.W)
        
        self.stress_result = tk.Label(stress_label_frame, text="--", 
                                     font=("Helvetica", 20, "bold"), bg="white", fg="#4CAF50")
        self.stress_result.pack(anchor=tk.W, pady=5)
        
        # All probabilities
        prob_frame = tk.LabelFrame(self.results_frame, text="Emotion Probabilities", 
                                   font=("Helvetica", 11, "bold"), bg="white", fg="#2c3e50")
        prob_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.prob_labels = {}
        emotions = ["happy", "neutral", "sad", "fear"]
        for emotion in emotions:
            frame = tk.Frame(prob_frame, bg="white")
            frame.pack(fill=tk.X, padx=5, pady=3)
            
            label = tk.Label(frame, text=f"{emotion.capitalize()}:", font=("Helvetica", 10), bg="white", width=12, anchor=tk.W)
            label.pack(side=tk.LEFT)
            
            canvas = tk.Canvas(frame, height=15, bg="#e0e0e0", highlightthickness=0)
            canvas.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            
            percent = tk.Label(frame, text="0%", font=("Helvetica", 9), bg="white", width=5)
            percent.pack(side=tk.LEFT)
            
            self.prob_labels[emotion] = {"canvas": canvas, "percent": percent}
        
        # Predict button
        self.predict_btn = tk.Button(self.results_frame, text="🔮 Predict Emotion & Stress", 
                                    command=self.predict,
                                    bg="#4CAF50", fg="white", font=("Helvetica", 11, "bold"),
                                    padx=15, pady=10)
        self.predict_btn.pack(pady=10)
        
    def upload_image(self):
        """Upload image from file"""
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        
        if file_path:
            self.load_image(file_path)
    
    def capture_from_camera(self):
        """Capture image from webcam"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open camera. Please check if camera is available.")
            return
        
        print("Camera opened. Press SPACE to capture, ESC to cancel")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Flip for selfie view
                frame = cv2.flip(frame, 1)
                
                # Add instructions
                cv2.putText(frame, "Press SPACE to capture, ESC to cancel", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow("Camera - Press SPACE to capture", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):
                    # Save captured frame
                    temp_path = "temp_capture.jpg"
                    cv2.imwrite(temp_path, frame)
                    self.load_image(temp_path)
                    break
                elif key == 27:  # ESC
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def load_image(self, image_path):
        """Load and display image"""
        try:
            # Load image
            self.original_image = cv2.imread(image_path)
            if self.original_image is None:
                messagebox.showerror("Error", "Could not load image")
                return
            
            # Store path
            self.selected_image = image_path
            
            # Display image (resize for display)
            display_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            display_image = cv2.resize(display_image, (300, 300))
            
            # Convert to PIL
            pil_image = Image.fromarray(display_image)
            photo = ImageTk.PhotoImage(pil_image)
            
            # Update canvas
            self.image_canvas.config(image=photo)
            self.image_canvas.image = photo
            
            # Clear previous results
            self.emotion_result.config(text="--")
            self.stress_result.config(text="--")
            self.confidence_text.config(text="0%")
            
            messagebox.showinfo("Success", "Image loaded successfully!\nClick 'Predict' to analyze.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {e}")
    
    def preprocess_image(self, image_path):
        """Preprocess image for model input"""
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("Could not read image")
            
            # Convert to grayscale
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Resize to 48x48
            img_resized = cv2.resize(img_gray, (self.IMG_SIZE, self.IMG_SIZE))
            
            # Normalize to [0, 1]
            img_normalized = img_resized.astype('float32') / 255.0
            
            # Add channel dimension
            img_expanded = np.expand_dims(img_normalized, -1)
            
            # Add batch dimension
            img_batch = np.expand_dims(img_expanded, 0)
            
            return img_batch
        except Exception as e:
            raise Exception(f"Image preprocessing error: {e}")
    
    def predict(self):
        """Predict emotion and stress level"""
        if self.selected_image is None:
            messagebox.showwarning("Warning", "Please load an image first!")
            return
        
        if self.model is None:
            messagebox.showerror("Error", "Model not loaded. Please check the model file.")
            return
        
        try:
            # Preprocess image
            img_batch = self.preprocess_image(self.selected_image)
            
            # Make prediction
            predictions = self.model.predict(img_batch, verbose=0)
            probabilities = predictions[0]
            
            # Get emotions in order
            emotions = ["happy", "neutral", "sad", "fear"]
            
            # Get top emotion
            top_idx = np.argmax(probabilities)
            top_emotion = emotions[top_idx]
            top_confidence = float(probabilities[top_idx]) * 100
            
            # Get stress level
            stress_level = self.stress_map[top_emotion]
            
            # Update UI
            self.emotion_result.config(text=top_emotion.upper(), 
                                      fg=self.emotion_colors[top_emotion])
            self.stress_result.config(text=stress_level)
            self.confidence_text.config(text=f"{top_confidence:.1f}%")
            
            # Update confidence bar
            self.update_confidence_bar(top_confidence)
            
            # Update probability bars
            self.update_probability_bars(probabilities, emotions)
            
            print(f"✓ Prediction: {top_emotion.upper()}")
            print(f"  Confidence: {top_confidence:.1f}%")
            print(f"  Stress Level: {stress_level}")
            print(f"  Probabilities: {dict(zip(emotions, (probabilities * 100).round(2)))}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {e}")
    
    def update_confidence_bar(self, confidence):
        """Update the confidence bar visualization"""
        self.confidence_bar.delete("all")
        
        # Draw background
        self.confidence_bar.create_rectangle(0, 0, self.confidence_bar.winfo_width(), 20, 
                                           fill="#e0e0e0", outline="none")
        
        # Draw confidence fill
        width = self.confidence_bar.winfo_width()
        fill_width = (confidence / 100) * width
        
        # Color based on confidence
        if confidence >= 80:
            color = "#4CAF50"  # Green
        elif confidence >= 60:
            color = "#FF9800"  # Orange
        else:
            color = "#F44336"  # Red
        
        self.confidence_bar.create_rectangle(0, 0, fill_width, 20, fill=color, outline="none")
    
    def update_probability_bars(self, probabilities, emotions):
        """Update probability bars for all emotions"""
        for i, emotion in enumerate(emotions):
            prob = float(probabilities[i]) * 100
            
            # Update canvas
            canvas = self.prob_labels[emotion]["canvas"]
            canvas.delete("all")
            
            width = canvas.winfo_width()
            fill_width = (prob / 100) * width
            
            canvas.create_rectangle(0, 0, width, 15, fill="#e0e0e0", outline="none")
            canvas.create_rectangle(0, 0, fill_width, 15, 
                                   fill=self.emotion_colors[emotion], outline="none")
            
            # Update percentage
            self.prob_labels[emotion]["percent"].config(text=f"{prob:.1f}%")

def main():
    root = tk.Tk()
    app = EmotionPredictorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
