import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import librosa
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
import sounddevice as sd
import soundfile as sf
import threading
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import warnings
warnings.filterwarnings('ignore')


class VoiceEmotionPredictorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Voice Emotion & Stress Level Predictor")
        self.root.geometry("900x700")
        self.root.resizable(True, True)
        
        # Configure style
        self.root.config(bg="#f0f0f0")
        
        # Audio parameters
        self.SAMPLE_RATE = 22050
        self.DURATION = 3
        self.recording = False
        self.audio_data = None
        
        # Model parameters
        self.model_path = r"ml models\cognitive_lstm_final_87.h5"
        self.model = None
        self.target_mfcc_length = 1 + int((self.SAMPLE_RATE * self.DURATION) / 512)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Emotion to stress mapping
        self.stress_map = {
            'neutral': 'Medium',
            'happy': 'Low',
            'sad': 'High',
            'angry': 'High'
        }
        
        # Emotion colors for visualization
        self.emotion_colors = {
            'happy': '#4CAF50',      # Green
            'neutral': '#2196F3',    # Blue
            'sad': '#FF9800',        # Orange
            'angry': '#F44336'       # Red
        }
        
        # Load model
        self.load_model()
        
        # Create UI
        self.create_ui()
    
    def load_model(self):
        """Load the trained LSTM model"""
        try:
            if os.path.exists(self.model_path):
                self.model = tf.keras.models.load_model(self.model_path)
                self.label_encoder.fit(['angry', 'happy', 'neutral', 'sad'])
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
        
        title_label = tk.Label(title_frame, text="🎤 Voice Emotion & Stress Level Predictor", 
                              font=("Helvetica", 18, "bold"), fg="white", bg="#2c3e50", pady=10)
        title_label.pack()
        
        # Main container
        main_frame = tk.Frame(self.root, bg="#f0f0f0")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left frame - Recording controls
        left_frame = tk.Frame(main_frame, bg="white")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        tk.Label(left_frame, text="Recording Controls", font=("Helvetica", 12, "bold"), bg="white").pack(pady=5)
        
        # Status label
        self.status_var = tk.StringVar(value="Ready to record...")
        status_label = tk.Label(left_frame, textvariable=self.status_var, 
                               font=("Helvetica", 10), bg="white", fg="#666666")
        status_label.pack(pady=5)
        
        # Recording info
        info_frame = tk.Frame(left_frame, bg="#e3f2fd")
        info_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(info_frame, text="📝 Recording Duration: 3 seconds", 
                font=("Helvetica", 10), bg="#e3f2fd").pack(pady=5)
        tk.Label(info_frame, text="Speak clearly and naturally", 
                font=("Helvetica", 9), bg="#e3f2fd", fg="#666666").pack(pady=3)
        
        # Button frame
        button_frame = tk.Frame(left_frame, bg="white")
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.record_btn = tk.Button(button_frame, text="🔴 Record Audio", 
                                    command=self.start_recording,
                                    bg="#F44336", fg="white", font=("Helvetica", 10, "bold"),
                                    padx=10, pady=8)
        self.record_btn.pack(side=tk.LEFT, padx=5)
        
        self.upload_btn = tk.Button(button_frame, text="📁 Upload Audio", 
                                    command=self.upload_audio,
                                    bg="#2196F3", fg="white", font=("Helvetica", 10, "bold"),
                                    padx=10, pady=8)
        self.upload_btn.pack(side=tk.LEFT, padx=5)
        
        self.clear_btn = tk.Button(button_frame, text="🗑 Clear", 
                                    command=self.clear_recording,
                                    bg="#9C27B0", fg="white", font=("Helvetica", 10, "bold"),
                                    padx=10, pady=8)
        self.clear_btn.pack(side=tk.LEFT, padx=5)
        
        # Waveform display
        waveform_label_frame = tk.Frame(left_frame, bg="white")
        waveform_label_frame.pack(fill=tk.X, padx=10, pady=(10, 5))
        tk.Label(waveform_label_frame, text="Audio Waveform", font=("Helvetica", 11, "bold"), bg="white").pack(anchor=tk.W)
        
        self.waveform_canvas = tk.Canvas(left_frame, height=100, bg="#f5f5f5", highlightthickness=1, highlightbackground="#ddd")
        self.waveform_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        # Right frame - Results
        right_frame = tk.Frame(main_frame, bg="white")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        tk.Label(right_frame, text="Prediction Results", font=("Helvetica", 12, "bold"), bg="white").pack(pady=5)
        
        # Results display
        self.results_frame = tk.Frame(right_frame, bg="white")
        self.results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Emotion label
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
        emotions = ['angry', 'happy', 'neutral', 'sad']
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
    
    def upload_audio(self):
        """Upload audio file"""
        file_path = filedialog.askopenfilename(
            title="Select an audio file",
            filetypes=[("Audio files", "*.wav *.mp3"), ("All files", "*.*")]
        )
        
        if file_path:
            self.load_audio(file_path)
    
    def load_audio(self, file_path):
        """Load and process audio file"""
        try:
            # Load audio
            self.audio_data, sr = librosa.load(file_path, sr=self.SAMPLE_RATE, duration=self.DURATION)
            
            # Store file path
            self.audio_file = file_path
            
            # Display waveform
            self.draw_waveform(self.audio_data)
            
            # Clear previous results
            self.emotion_result.config(text="--")
            self.stress_result.config(text="--")
            self.confidence_text.config(text="0%")
            
            self.status_var.set("Audio loaded successfully! Click 'Predict' to analyze.")
            messagebox.showinfo("Success", "Audio loaded successfully!\nClick 'Predict' to analyze.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load audio: {e}")
            self.status_var.set("Error loading audio")
    
    def start_recording(self):
        """Start recording audio"""
        self.recording = True
        self.audio_data = None
        self.record_btn.config(state=tk.DISABLED)
        self.upload_btn.config(state=tk.DISABLED)
        self.predict_btn.config(state=tk.DISABLED)
        self.status_var.set("Recording... Please speak now")
        self.waveform_canvas.delete("all")
        
        # Record in a separate thread
        thread = threading.Thread(target=self._record_audio)
        thread.daemon = True
        thread.start()
    
    def _record_audio(self):
        """Record audio data"""
        try:
            frames_count = int(self.SAMPLE_RATE * self.DURATION)
            audio = sd.rec(frames_count, samplerate=self.SAMPLE_RATE, channels=1, dtype=np.float32)
            sd.wait()
            
            self.audio_data = audio.flatten()
            self.recording = False
            self.status_var.set("Recording complete! Click 'Predict' to analyze.")
            
            # Draw waveform
            self.draw_waveform(self.audio_data)
            
            self.record_btn.config(state=tk.NORMAL)
            self.upload_btn.config(state=tk.NORMAL)
            self.predict_btn.config(state=tk.NORMAL)
            
        except Exception as e:
            self.status_var.set(f"Recording error: {str(e)}")
            self.recording = False
            self.record_btn.config(state=tk.NORMAL)
            self.upload_btn.config(state=tk.NORMAL)
    
    def draw_waveform(self, audio):
        """Draw audio waveform"""
        try:
            self.waveform_canvas.delete("all")
            
            if len(audio) == 0:
                return
            
            canvas_width = self.waveform_canvas.winfo_width()
            canvas_height = self.waveform_canvas.winfo_height()
            
            if canvas_width <= 1 or canvas_height <= 1:
                canvas_width, canvas_height = 800, 100
            
            # Normalize audio
            audio_norm = audio / (np.max(np.abs(audio)) + 1e-8)
            
            # Downsample for display
            step = len(audio) // canvas_width
            if step < 1:
                step = 1
            
            audio_display = audio_norm[::step]
            
            # Draw waveform
            for i, sample in enumerate(audio_display[:-1]):
                x1 = i
                y1 = canvas_height / 2 - (sample * canvas_height / 2)
                x2 = i + 1
                y2 = canvas_height / 2 - (audio_display[i + 1] * canvas_height / 2)
                
                self.waveform_canvas.create_line(x1, y1, x2, y2, fill="#2196F3", width=1)
            
        except Exception as e:
            print(f"Waveform drawing error: {e}")
    
    def clear_recording(self):
        """Clear the recording"""
        self.audio_data = None
        self.waveform_canvas.delete("all")
        self.emotion_result.config(text="--")
        self.stress_result.config(text="--")
        self.confidence_text.config(text="0%")
        self.status_var.set("Cleared. Ready to record...")
        
        # Clear probabilities
        for emotion in self.prob_labels:
            self.prob_labels[emotion]["canvas"].delete("all")
            self.prob_labels[emotion]["percent"].config(text="0%")
    
    def extract_mfcc(self, audio):
        """Extract MFCC features from audio"""
        try:
            # Extract MFCCs
            mfccs = librosa.feature.mfcc(y=audio, sr=self.SAMPLE_RATE, n_mfcc=40)
            
            # Ensure consistent length
            if mfccs.shape[1] < self.target_mfcc_length:
                pad_width = self.target_mfcc_length - mfccs.shape[1]
                mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
            elif mfccs.shape[1] > self.target_mfcc_length:
                mfccs = mfccs[:, :self.target_mfcc_length]
            
            return mfccs
        except Exception as e:
            raise Exception(f"MFCC extraction failed: {str(e)}")
    
    def predict(self):
        """Predict emotion and stress level"""
        if self.audio_data is None or len(self.audio_data) == 0:
            messagebox.showwarning("Warning", "Please record or upload audio first!")
            return
        
        if self.model is None:
            messagebox.showerror("Error", "Model not loaded. Please check the model file.")
            return
        
        try:
            # Extract MFCC features
            mfcc = self.extract_mfcc(self.audio_data)
            
            # Normalize
            mfcc_flat = mfcc.flatten().reshape(1, -1)
            mfcc_scaled = self.scaler.fit_transform(mfcc_flat)
            mfcc_reshaped = mfcc_scaled.reshape(1, 40, self.target_mfcc_length)
            
            # Make prediction
            predictions = self.model.predict(mfcc_reshaped, verbose=0)
            probabilities = predictions[0]
            
            # Get emotion
            emotion_idx = np.argmax(probabilities)
            emotion = self.label_encoder.classes_[emotion_idx]
            confidence = float(probabilities[emotion_idx])
            stress_level = self.stress_map.get(emotion, "Unknown")
            
            # Update results
            self.emotion_result.config(text=emotion.upper(), fg=self.emotion_colors.get(emotion, "#2196F3"))
            self.stress_result.config(text=stress_level)
            self.confidence_text.config(text=f"{confidence*100:.2f}%")
            
            # Update confidence bar
            self.confidence_bar.delete("all")
            bar_width = self.confidence_bar.winfo_width()
            if bar_width <= 1:
                bar_width = 300
            
            fill_width = bar_width * confidence
            self.confidence_bar.create_rectangle(0, 0, fill_width, 20, 
                                               fill=self.emotion_colors.get(emotion, "#2196F3"),
                                               outline=self.emotion_colors.get(emotion, "#2196F3"))
            
            # Update probability bars
            for emotion_name in self.label_encoder.classes_:
                idx = list(self.label_encoder.classes_).index(emotion_name)
                prob = float(probabilities[idx])
                
                canvas = self.prob_labels[emotion_name]["canvas"]
                percent_label = self.prob_labels[emotion_name]["percent"]
                
                canvas.delete("all")
                canvas_width = canvas.winfo_width()
                if canvas_width <= 1:
                    canvas_width = 200
                
                fill_width = canvas_width * prob
                canvas.create_rectangle(0, 0, fill_width, 15,
                                       fill=self.emotion_colors.get(emotion_name, "#999999"),
                                       outline=self.emotion_colors.get(emotion_name, "#999999"))
                
                percent_label.config(text=f"{prob*100:.1f}%")
            
            self.status_var.set("Prediction complete!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {e}")
            self.status_var.set("Error during prediction")


def main():
    root = tk.Tk()
    app = VoiceEmotionPredictorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
