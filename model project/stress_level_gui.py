import tkinter as tk
from tkinter import messagebox, ttk
import tkinter.font as tkFont
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class StressLevelPredictor:
    def __init__(self, root):
        self.root = root
        self.root.title("Stress Level Predictor Survey")
        self.root.geometry("900x800")
        self.root.resizable(True, True)
        
        # Configure style
        self.root.config(bg="#f0f0f0")
        
        # Model parameters
        self.model = None
        self.scaler = None
        self.feature_names = ['blood_pressure', 'sleep_quality', 'academic_performance', 
                             'teacher_student_relationship', 'basic_needs']
        
        # Stress level mapping
        self.stress_levels = {0: 'Low Stress', 1: 'Medium Stress', 2: 'High Stress'}
        self.stress_colors = {0: '#4CAF50', 1: '#FF9800', 2: '#F44336'}
        
        # Load model
        self.load_model()
        
        # Create UI
        self.create_ui()
    
    def load_model(self):
        """Load the trained model and scaler"""
        try:
            model_path = r"ml models\stress_model.pkl"
            scaler_path = r"ml models\scaler.pkl"
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                print(f"✓ Model loaded from {model_path}")
            else:
                messagebox.showerror("Error", "Model files not found!\nPlease train the model first.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
    
    def create_ui(self):
        """Create the user interface"""
        # Title
        title_frame = tk.Frame(self.root, bg="#2c3e50")
        title_frame.pack(fill=tk.X)
        
        title_label = tk.Label(title_frame, text="📋 Stress Level Predictor Survey", 
                              font=("Helvetica", 18, "bold"), fg="white", bg="#2c3e50", pady=10)
        title_label.pack()
        
        # Main container with scrollbar
        main_frame = tk.Frame(self.root, bg="#f0f0f0")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create canvas with scrollbar
        canvas = tk.Canvas(main_frame, bg="#f0f0f0", highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="#f0f0f0")
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Instructions
        instructions = tk.Label(scrollable_frame, 
                               text="Please answer the following questions (0-5 scale) to predict your stress level",
                               font=("Helvetica", 11), bg="#f0f0f0", fg="#666666", wraplength=700)
        instructions.pack(pady=10)
        
        # Questions Frame
        questions_frame = tk.Frame(scrollable_frame, bg="white")
        questions_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=10)
        
        # Question 1: Blood Pressure
        q1_frame = self.create_question_frame(questions_frame, 
                                             "1. How would you rate your blood pressure?",
                                             ["0 - Very Low", "1 - Low", "2 - Normal", "3 - High", "4 - Very High"],
                                             "bp_var")
        
        # Question 2: Sleep Quality
        q2_frame = self.create_question_frame(questions_frame,
                                             "2. How would you rate your sleep quality?",
                                             ["0 - Very Poor", "1 - Poor", "2 - Average", "3 - Good", "4 - Very Good"],
                                             "sleep_var")
        
        # Question 3: Academic Performance
        q3_frame = self.create_question_frame(questions_frame,
                                             "3. How would you rate your academic performance?",
                                             ["0 - Very Poor", "1 - Poor", "2 - Average", "3 - Good", "4 - Very Good"],
                                             "academic_var")
        
        # Question 4: Teacher-Student Relationship
        q4_frame = self.create_question_frame(questions_frame,
                                             "4. How would you rate your teacher-student relationship?",
                                             ["0 - Very Poor", "1 - Poor", "2 - Average", "3 - Good", "4 - Very Good"],
                                             "teacher_var")
        
        # Question 5: Basic Needs
        q5_frame = self.create_question_frame(questions_frame,
                                             "5. How well are your basic needs met?",
                                             ["0 - Not Met", "1 - Barely Met", "2 - Partially Met", "3 - Well Met", "4 - Fully Met"],
                                             "basic_var")
        
        # Pack scrollbar and canvas
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Bottom frame for buttons and results
        bottom_frame = tk.Frame(self.root, bg="#f0f0f0")
        bottom_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Button frame
        button_frame = tk.Frame(bottom_frame, bg="white")
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.predict_btn = tk.Button(button_frame, text="🔮 Predict Stress Level", 
                                    command=self.predict,
                                    bg="#4CAF50", fg="white", font=("Helvetica", 12, "bold"),
                                    padx=20, pady=10)
        self.predict_btn.pack(side=tk.LEFT, padx=5)
        
        self.clear_btn = tk.Button(button_frame, text="🗑 Clear", 
                                  command=self.clear_form,
                                  bg="#9C27B0", fg="white", font=("Helvetica", 12, "bold"),
                                  padx=20, pady=10)
        self.clear_btn.pack(side=tk.LEFT, padx=5)
        
        # Results frame
        results_frame = ttk.LabelFrame(bottom_frame, text="Prediction Results", padding=15)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Stress level result
        stress_label_frame = tk.Frame(results_frame, bg="white")
        stress_label_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(stress_label_frame, text="Predicted Stress Level:", 
                font=("Helvetica", 11, "bold"), bg="white").pack(anchor=tk.W)
        
        self.stress_result = tk.Label(stress_label_frame, text="--", 
                                     font=("Helvetica", 20, "bold"), bg="white", fg="#4CAF50")
        self.stress_result.pack(anchor=tk.W, pady=5)
        
        # Confidence display
        confidence_frame = tk.Frame(results_frame, bg="white")
        confidence_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(confidence_frame, text="Confidence Scores:", 
                font=("Helvetica", 11, "bold"), bg="white").pack(anchor=tk.W)
        
        self.confidence_text = tk.Text(confidence_frame, height=4, width=60, 
                                      font=("Courier", 10), bg="#f5f5f5", fg="#333333",
                                      relief=tk.SUNKEN, bd=1)
        self.confidence_text.pack(fill=tk.BOTH, expand=True, pady=5)
        self.confidence_text.config(state=tk.DISABLED)
    
    def create_question_frame(self, parent, question_text, options, var_name):
        """Create a question frame with radio buttons"""
        frame = tk.LabelFrame(parent, text=question_text, font=("Helvetica", 10, "bold"),
                             bg="white", fg="#333333", padx=10, pady=10)
        frame.pack(fill=tk.X, padx=5, pady=8)
        
        var = tk.StringVar(value=options[2])  # Default to middle option
        setattr(self, var_name, var)
        
        options_frame = tk.Frame(frame, bg="white")
        options_frame.pack(fill=tk.X)
        
        for option in options:
            rb = tk.Radiobutton(options_frame, text=option, variable=var, value=option,
                              font=("Helvetica", 10), bg="white", fg="#333333",
                              activebackground="#f0f0f0", activeforeground="#2196F3")
            rb.pack(anchor=tk.W, pady=3)
        
        return frame
    
    def extract_values(self):
        """Extract numerical values from radio button selections"""
        try:
            bp = int(self.bp_var.get().split()[0])
            sleep = int(self.sleep_var.get().split()[0])
            academic = int(self.academic_var.get().split()[0])
            teacher = int(self.teacher_var.get().split()[0])
            basic = int(self.basic_var.get().split()[0])
            
            return [bp, sleep, academic, teacher, basic]
        except Exception as e:
            messagebox.showerror("Error", f"Failed to extract values: {e}")
            return None
    
    def predict(self):
        """Predict stress level"""
        if self.model is None or self.scaler is None:
            messagebox.showerror("Error", "Model not loaded!")
            return
        
        try:
            # Extract values
            values = self.extract_values()
            if values is None:
                return
            
            # Convert to numpy array and reshape
            X = np.array(values).reshape(1, -1)
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Make prediction
            prediction = self.model.predict(X_scaled)[0]
            probabilities = self.model.predict_proba(X_scaled)[0]
            
            # Get stress level
            stress_level = self.stress_levels[prediction]
            stress_color = self.stress_colors[prediction]
            
            # Update results
            self.stress_result.config(text=stress_level, fg=stress_color)
            
            # Display confidence scores
            self.confidence_text.config(state=tk.NORMAL)
            self.confidence_text.delete(1.0, tk.END)
            
            results_text = "━" * 50 + "\n"
            results_text += "CONFIDENCE BREAKDOWN\n"
            results_text += "━" * 50 + "\n\n"
            
            for i, stress_type in enumerate(self.stress_levels.values()):
                confidence = probabilities[i]
                bar_length = int(confidence * 30)
                bar = "█" * bar_length + "░" * (30 - bar_length)
                results_text += f"{stress_type:15s} │{bar}│ {confidence*100:6.2f}%\n"
            
            results_text += "\n" + "━" * 50
            
            self.confidence_text.insert(1.0, results_text)
            self.confidence_text.config(state=tk.DISABLED)
            
            messagebox.showinfo("Success", f"Predicted Stress Level: {stress_level}\n\nConfidence: {probabilities[prediction]*100:.2f}%")
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
    
    def clear_form(self):
        """Clear all selections"""
        self.bp_var.set("2 - Normal")
        self.sleep_var.set("2 - Average")
        self.academic_var.set("2 - Average")
        self.teacher_var.set("2 - Average")
        self.basic_var.set("2 - Partially Met")
        
        self.stress_result.config(text="--", fg="#4CAF50")
        self.confidence_text.config(state=tk.NORMAL)
        self.confidence_text.delete(1.0, tk.END)
        self.confidence_text.config(state=tk.DISABLED)


def main():
    root = tk.Tk()
    app = StressLevelPredictor(root)
    root.mainloop()


if __name__ == "__main__":
    main()
