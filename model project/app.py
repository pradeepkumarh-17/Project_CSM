from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS
from dotenv import load_dotenv

# Load .env from project root if present
load_dotenv()
from pathlib import Path
import numpy as np
import cv2
import librosa
import joblib
import traceback
import os
import subprocess
import tempfile
import shutil
import logging

from datetime import datetime
import json


# Gemini fallback removed: external generative-AI fallback disabled due to OAuth2-based auth requirements.
GEMINI_API_KEY = None
GEMINI_API_URL = None
logging.info("Gemini fallback disabled for this deployment")


# Detect optional audio conversion tools
HAS_PYDUB = False
AudioSegment = None
try:
    from pydub import AudioSegment as _AudioSegment  # optional dependency
    HAS_PYDUB = True
    AudioSegment = _AudioSegment
except Exception:
    HAS_PYDUB = False

# Check ffmpeg availability in PATH
HAS_FFMPEG = shutil.which("ffmpeg") is not None
# Allow overriding/pointing to a custom ffmpeg via environment variable
FFMPEG_PATH = os.environ.get("FFMPEG_PATH") or os.environ.get("FFMPEG_BINARY") or os.environ.get("FFMPEG_DIR")
configured_ffmpeg = None
configured_ffprobe = None

# Helper to probe a candidate path
def _probe_ffmpeg_candidate(ff_path):
    p = Path(ff_path)
    if p.is_file():
        return str(p)
    candidate = p / "bin" / "ffmpeg.exe"
    if candidate.exists():
        return str(candidate)
    candidate2 = p / "ffmpeg.exe"
    if candidate2.exists():
        return str(candidate2)
    return None

# 1) If user set an env var, try that first
if FFMPEG_PATH:
    try:
        cand = _probe_ffmpeg_candidate(FFMPEG_PATH)
        if cand:
            configured_ffmpeg = cand
            configured_ffprobe = cand.replace("ffmpeg", "ffprobe")
            HAS_FFMPEG = True
    except Exception:
        pass

# 2) Auto-discover common locations in the workspace and system (e.g., C:\ffmpeg or E:\Star\ffmpeg*)
if not configured_ffmpeg:
    try:
        workspace_root = Path(__file__).resolve().parents[1]  # project parent (e.g., E:\Star)
        for candidate_dir in list(workspace_root.glob("ffmpeg*"))[:5]:
            cand = _probe_ffmpeg_candidate(candidate_dir)
            if cand:
                configured_ffmpeg = cand
                configured_ffprobe = cand.replace("ffmpeg", "ffprobe")
                HAS_FFMPEG = True
                break
    except Exception:
        pass

    # Check the conventional location C:\ffmpeg
    if not configured_ffmpeg:
        cand = _probe_ffmpeg_candidate(Path("C:/ffmpeg"))
        if cand:
            configured_ffmpeg = cand
            configured_ffprobe = cand.replace("ffmpeg", "ffprobe")
            HAS_FFMPEG = True

# Ensure ffprobe is discovered when ffmpeg is in PATH
if HAS_FFMPEG and not configured_ffprobe:
    probe = shutil.which("ffprobe")
    if probe:
        configured_ffprobe = probe

# If we found a configured ffmpeg, add its directory to PATH so subprocess/pydub can find it
if configured_ffmpeg:
    ff_dir = str(Path(configured_ffmpeg).parent)
    if ff_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = ff_dir + os.pathsep + os.environ.get("PATH", "")
        logging.info(f"Added ffmpeg dir to PATH: {ff_dir}")

# If pydub is available, set its explicit converter and ffprobe paths where possible
if HAS_PYDUB:
    try:
        from pydub.utils import which as pydub_which
        # Prefer an explicitly configured path then fall back to pydub.which()
        ffmpeg_candidate = configured_ffmpeg or pydub_which("ffmpeg")
        ffprobe_candidate = configured_ffprobe or pydub_which("ffprobe")

        if ffmpeg_candidate:
            AudioSegment.converter = str(ffmpeg_candidate)
            configured_ffmpeg = str(ffmpeg_candidate)
            HAS_FFMPEG = True
            ff_dir = str(Path(configured_ffmpeg).parent)
            if ff_dir not in os.environ.get("PATH", ""):
                os.environ["PATH"] = ff_dir + os.pathsep + os.environ.get("PATH", "")
            logging.info(f"Using ffmpeg: {configured_ffmpeg}")
        else:
            logging.warning("ffmpeg not found via FFMPEG_PATH or pydub.which(); conversions may fail")

        if ffprobe_candidate:
            AudioSegment.ffprobe = str(ffprobe_candidate)
            configured_ffprobe = str(ffprobe_candidate)
            logging.info(f"Using ffprobe: {configured_ffprobe}")

    except Exception:
        logging.exception("Failed to set pydub converter/ffprobe explicitly")

# =====================================================
# App & CORS
# =====================================================
# By serving the frontend and backend from the same origin (port 5000),
# we avoid cross-origin (CORS) issues and simplify the setup.
# The frontend files will be accessible directly, e.g., http://127.0.0.1:5000/index.html
STATIC_DIR = Path(__file__).resolve().parent.parent / "frontend"
app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="")
CORS(app) # Keep CORS for cases where it's still needed (e.g., other clients)

# Database & auth support will be initialized in the db package
# Initialize DB, migrations and auth blueprints
from db import init_app as init_db
from db import models  # ensures models are registered
from db.auth_routes import bp as auth_bp

# Initialize with app config / env vars
init_db(app)
app.register_blueprint(auth_bp)
logging.info("Database and auth support enabled")

# Development convenience: create tables if they are missing. In production prefer Flask-Migrate.
from db import db
with app.app_context():
    db.create_all()




# =====================================================
# Paths & Config
# =====================================================
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

IMG_SIZE = 48

ALLOWED_IMAGE_EXT = {"png", "jpg", "jpeg", "bmp", "gif"}
ALLOWED_AUDIO_EXT = {"wav", "mp3", "m4a", "ogg", "flac", "webm"}

FACE_MODEL_PATH = BASE_DIR / "ml models" / "fer4_cnn.h5"
VOICE_MODEL_PATH = BASE_DIR / "ml models" / "cognitive_lstm_final_87.h5"
SURVEY_MODEL_PATH = BASE_DIR / "ml models" / "stress_model.pkl"
SCALER_PATH = BASE_DIR / "ml models" / "scaler.pkl"

import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# ===============================
# Load models at startup
# ===============================
model_errors = {}

try:
    face_model = tf.keras.models.load_model(FACE_MODEL_PATH)
    print("[OK] Face model loaded at startup")
except Exception as e:
    face_model = None
    model_errors["face"] = str(e)

try:
    voice_model = tf.keras.models.load_model(VOICE_MODEL_PATH)
    print("[OK] Voice model loaded at startup")
except Exception as e:
    voice_model = None
    model_errors["voice"] = str(e)

try:
    survey_model = joblib.load(SURVEY_MODEL_PATH)
    print("[OK] Survey model loaded at startup")
except Exception as e:
    survey_model = None
    model_errors["survey"] = str(e)

try:
    scaler = joblib.load(SCALER_PATH)
    print("[OK] Scaler loaded at startup")
except Exception as e:
    scaler = None
    model_errors["scaler"] = str(e)

# Optional: voice scaler trained during model prep may be saved as voice_scaler.pkl
VOICE_SCALER_PATH = BASE_DIR / "ml models" / "voice_scaler.pkl"
try:
    voice_scaler = joblib.load(VOICE_SCALER_PATH) if VOICE_SCALER_PATH.exists() else None
    if voice_scaler is not None:
        print("[OK] Voice scaler loaded at startup")
    else:
        voice_scaler = None
except Exception as e:
    voice_scaler = None
    model_errors["voice_scaler"] = str(e)

# =====================================================
# Mappings
# =====================================================
FACE_EMOTIONS = ["happy", "neutral", "sad", "fear"]
VOICE_EMOTIONS = ["neutral", "happy", "sad", "angry"]

STRESS_MAP_FACE = {
    "happy": "Low",
    "neutral": "Medium",
    "sad": "High",
    "fear": "High"
}

STRESS_MAP_VOICE = {
    "happy": "Low",
    "neutral": "Medium",
    "sad": "High",
    "angry": "High"
}

STRESS_MAP_SURVEY = {0: "Low", 1: "Medium", 2: "High"}

# =====================================================
# Utilities
# =====================================================
def allowed_file(filename, allowed):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed


def preprocess_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    normalized = resized / 255.0
    return normalized.reshape(1, IMG_SIZE, IMG_SIZE, 1)


def _convert_to_wav_if_needed(path, target_sr=22050):
    """
    Convert non-wav audio files (e.g., .webm) to a temporary WAV file.
    Uses pydub (if installed) or falls back to ffmpeg CLI.
    Returns (converted_path, created_temp_bool).
    """
    ext = Path(path).suffix.lower()
    if ext == ".wav":
        return path, False

    # Try pydub first (if available)
    if HAS_PYDUB:
        try:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.close()
            audio = AudioSegment.from_file(path)
            audio = audio.set_frame_rate(target_sr).set_channels(1)
            audio.export(tmp.name, format="wav")
            return tmp.name, True
        except Exception as exc:
            logging.exception("pydub failed to convert audio")
            # fall through to ffmpeg if available

    # Fallback to ffmpeg if available
    if HAS_FFMPEG:
        try:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.close()
            ffmpeg_exec = configured_ffmpeg if configured_ffmpeg else "ffmpeg"
            cmd = [ffmpeg_exec, "-y", "-i", str(path), "-ar", str(target_sr), "-ac", "1", str(tmp.name)]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return tmp.name, True
        except FileNotFoundError as fnf:
            logging.exception("ffmpeg executable not found when attempting conversion")
            raise RuntimeError("ffmpeg executable not found. Ensure ffmpeg is installed and in PATH, or set FFMPEG_PATH to the ffmpeg executable.") from fnf
        except Exception as exc:
            logging.exception("ffmpeg failed to convert audio")
            raise RuntimeError("Audio conversion failed using ffmpeg.") from exc

    # Neither tool is available
    raise RuntimeError(
        "Could not convert audio to WAV. Please install 'pydub' (`pip install pydub`) and ensure ffmpeg is installed and on PATH.\n"
        "On Windows you can install ffmpeg via Chocolatey: `choco install ffmpeg -y`, or download from https://ffmpeg.org/download.html and add to PATH."
    )


# Gemini fallback removed. External generative-AI fallback is not used due to OAuth2 authentication requirements.


def extract_voice_features(wav_path, target_sr=22050, duration=3, target_frames=None):
    """
    Produce stacked features matching training: MFCC (40) + delta (40) + delta-delta (40) + 10x ZCR = 130 dims
    Returns shape (1, timesteps, features)
    Pads or truncates along the time axis to `target_frames` (falls back to model input or 130 frames).
    """
    converted_path, created = _convert_to_wav_if_needed(wav_path, target_sr=target_sr)
    try:
        y, sr = librosa.load(converted_path, sr=target_sr, duration=duration)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        zcr = librosa.feature.zero_crossing_rate(y)
        # Expand ZCR to 10 rows to reach ~130 feature dims (10 x ZCR)
        zcr = np.repeat(zcr, 10, axis=0)

        # Stack features -> shape (130, frames)
        stacked = np.vstack([mfcc, delta, delta2, zcr])

        # Determine desired frames: prefer explicit target_frames, else model input, else 130
        desired_frames = int(target_frames) if target_frames else None
        if desired_frames is None:
            try:
                desired_frames = int(voice_model.input_shape[1]) if (voice_model is not None and hasattr(voice_model, 'input_shape')) else 130
            except Exception:
                desired_frames = 130

        frames = stacked.shape[1]
        if frames < desired_frames:
            pad_width = desired_frames - frames
            stacked = np.hstack([stacked, np.zeros((stacked.shape[0], pad_width), dtype=stacked.dtype)])
        elif frames > desired_frames:
            stacked = stacked[:, :desired_frames]

        features = stacked.T  # (timesteps, features)

        # Diagnostics
        try:
            logging.info(f"[VOICE] extract_voice_features: frames={frames}->{desired_frames}, shape={features.shape}, min={features.min():.6f}, max={features.max():.6f}, mean={features.mean():.6f}")
            if np.allclose(features, 0):
                logging.warning("[VOICE] extracted features are all zeros (possible conversion/load error)")
        except Exception:
            pass

        return features.reshape(1, features.shape[0], features.shape[1]).astype(np.float32)
    finally:
        if created:
            try:
                Path(converted_path).unlink(missing_ok=True)
            except Exception:
                pass


# Startup loading handled earlier

# =====================================================
# Routes
# =====================================================
@app.route("/")
def index():
    """Serves the main index.html file from the frontend directory."""
    return app.send_static_file("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "face_model_loaded": face_model is not None,
        "voice_model_loaded": voice_model is not None,
        "survey_model_loaded": survey_model is not None,
        "scaler_loaded": scaler is not None,
        "voice_scaler_loaded": (voice_scaler is not None),
        "pydub_installed": HAS_PYDUB,
        "ffmpeg_installed": HAS_FFMPEG,
        "ffmpeg_configured_path": configured_ffmpeg,
        "ffprobe_configured_path": configured_ffprobe,
        "pydub_converter": getattr(AudioSegment, "converter", None) if HAS_PYDUB else None,
        "pydub_ffprobe": getattr(AudioSegment, "ffprobe", None) if HAS_PYDUB else None,
        "gemini_configured": False,
        "gemini_url": None,
        "errors": model_errors
    })


# -----------------------------
# Helper: save assessment for current user (DB removed; no-op)
# -----------------------------
def save_assessment_entry(face=None, voice=None, survey=None, overall=None, metadata=None):
    """Persist an assessment record to the database and return its id (or None on failure).

    Note: This will only save if the current user is authenticated (user_id required by schema).
    """
    try:
        from db.models import Assessment, User
        from db import db
        from flask_login import current_user

        user_id = None
        if not (current_user and getattr(current_user, 'is_authenticated', False)):
            # Fallback: Use a default guest user so data is saved even without login
            guest = User.query.filter_by(email='guest@localhost').first()
            if not guest:
                guest = User(email='guest@localhost', name='Guest')
                guest.set_password('guest')
                db.session.add(guest)
                db.session.commit()
            user_id = guest.id
        else:
            user_id = current_user.id
            
        entry = Assessment(user_id=user_id, stress_level=overall, assessment_data=metadata)
        db.session.add(entry)
        db.session.commit()
        logging.debug(f"Saved assessment id={entry.id} user_id={user_id}")
        return entry.id
    except Exception:
        logging.exception("Failed to save assessment entry")
        try:
            from db import db
            db.session.rollback()
        except Exception:
            pass
        return None



# Auth endpoints and DB-backed assessment endpoints removed (DB disabled)



# -------------------- FACE ----------------------------
@app.route("/predict_face", methods=["POST"])
def predict_face():
    global face_model
    try:
        if "image" not in request.files:
            raise ValueError("Image not provided")

        file = request.files["image"]
        if not allowed_file(file.filename, ALLOWED_IMAGE_EXT):
            raise ValueError("Invalid image format")

        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        img_input = preprocess_face(img)

        if face_model is None:
            raise ValueError("Face model not loaded")

        probs = face_model.predict(img_input, verbose=0)[0]
        idx = int(np.argmax(probs))

        resp = dict(
            success=True,
            emotion=FACE_EMOTIONS[idx],
            stressLevel=STRESS_MAP_FACE[FACE_EMOTIONS[idx]],
            confidence=round(float(np.max(probs) * 100), 2)
        )
        try:
            save_assessment_entry(face={'label': resp['emotion'], 'conf': resp['confidence']}, overall=resp['stressLevel'])
        except Exception:
            logging.exception('Failed to persist face assessment')
        return jsonify(resp)

    except Exception as e:
        traceback.print_exc()
        return jsonify(success=False, error=str(e)), 500


# -------------------- VOICE ---------------------------
@app.route("/predict_voice", methods=["POST"])
def predict_voice():
    global voice_model
    try:
        if "audio" not in request.files:
            raise ValueError("Audio not provided")

        file = request.files["audio"]
        if not allowed_file(file.filename, ALLOWED_AUDIO_EXT):
            raise ValueError("Invalid audio format")

        temp_path = UPLOAD_DIR / secure_filename(file.filename)
        file.save(temp_path)

        features = extract_voice_features(str(temp_path))
        # Keep temp file until fallback is attempted in case of errors

        # Prefer a saved voice scaler if available (same preprocessing used during training).
        try:
            if 'voice_scaler' in globals() and voice_scaler is not None:
                flat = features.reshape(1, -1)
                flat_scaled = voice_scaler.transform(flat)
                features = flat_scaled.reshape(features.shape)
                logging.info("[VOICE] Applied saved voice scaler to features")
            else:
                # Per-sample per-feature normalization (normalize each feature column across time)
                arr = features[0]  # (timesteps, features)
                mean = arr.mean(axis=0, keepdims=True)
                std = arr.std(axis=0, keepdims=True)
                std[std < 1e-6] = 1.0
                arr = (arr - mean) / std
                features = arr.reshape(1, arr.shape[0], arr.shape[1]).astype(np.float32)
                logging.info("[VOICE] Applied per-sample feature normalization")

            logging.debug(f"[VOICE] post-scale stats min={features.min():.6f}, max={features.max():.6f}, mean={features.mean():.6f}, shape={features.shape}")
        except Exception:
            logging.exception("Voice feature scaling/normalization failed; proceeding without further scaling")

        if voice_model is None:
            raise ValueError("Voice model not loaded")

        preds = voice_model.predict(features, verbose=0)
        logging.info(f"[VOICE] model raw output shape: {preds.shape}")
        # Handle models that might emit per-frame predictions (e.g., (1, frames, n_classes)) or per-sample (1, n_classes)
        if preds.ndim == 3:
            probs = preds.mean(axis=1)[0]
        else:
            probs = preds[0]
        logging.info(f"[VOICE] probabilities: {probs}")
        idx = int(np.argmax(probs))

        # Clean up uploaded file
        temp_path.unlink(missing_ok=True)

        resp = dict(
            success=True,
            emotion=VOICE_EMOTIONS[idx],
            stressLevel=STRESS_MAP_VOICE[VOICE_EMOTIONS[idx]],
            confidence=round(float(np.max(probs) * 100), 2),
            source="model"
        )
        try:
            save_assessment_entry(voice={'label': resp['emotion'], 'conf': resp['confidence']}, overall=resp['stressLevel'])
        except Exception:
            logging.exception('Failed to persist voice assessment')
        # Clean up uploaded file
        temp_path.unlink(missing_ok=True)
        return jsonify(resp)

    except Exception as e:
        # Log the exception and return a safe default response (no external fallback used)
        logging.exception("Voice prediction failed; returning safe default response: %s", e)
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            pass

        return jsonify(
            success=True,
            emotion="neutral",
            stressLevel="Medium",
            confidence=50.0,
            source="default"
        )


# -------------------- SURVEY --------------------------
@app.route("/predict_survey", methods=["POST"])
def predict_survey():
    try:
        if survey_model is None:
            raise ValueError("Survey model is not loaded")
        if scaler is None:
            raise ValueError("Scaler is not loaded")

        data = request.get_json()
        answers = data.get("answers")

        if not answers:
            raise ValueError("Survey answers missing")

        # The model expects features in a specific order. This list was derived
        # from the model training script and the associated GUI.
        feature_names = [
            'blood_pressure', 'sleep_quality', 'academic_performance',
            'teacher_student_relationship', 'basic_needs'
        ]

        # Extract values from answers dict in the correct order.
        # This now assumes the frontend sends a dictionary with these exact keys.
        # e.g., {"blood_pressure": 1, "sleep_quality": 2, ...}
        input_values = [answers.get(feature) for feature in feature_names]

        if any(v is None for v in input_values):
            missing = [name for name, val in zip(feature_names, input_values) if val is None]
            raise ValueError(f"Missing required survey answers: {', '.join(missing)}")

        # Convert to numpy array, reshape, and scale
        X = np.array(input_values).reshape(1, -1)
        X_scaled = scaler.transform(X)

        # Make prediction
        prediction = survey_model.predict(X_scaled)[0]
        probabilities = survey_model.predict_proba(X_scaled)[0]

        stress = STRESS_MAP_SURVEY[prediction]
        confidence = round(float(probabilities[prediction] * 100), 2)

        resp = dict(
            success=True,
            stressLevel=stress,
            confidence=confidence,
            probabilities={STRESS_MAP_SURVEY[i]: round(float(p * 100), 2) for i, p in enumerate(probabilities)}
        )
        try:
            save_assessment_entry(survey={'label': resp['stressLevel'], 'conf': resp['confidence']}, overall=resp['stressLevel'])
        except Exception:
            logging.exception('Failed to persist survey assessment')
        return jsonify(resp)

    except Exception as e:
        traceback.print_exc()
        return jsonify(success=False, error=str(e)), 500


# =====================================================
# Main
# =====================================================
if __name__ == "__main__":
    print("Emotion Predictor Server Running")
    print("http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
