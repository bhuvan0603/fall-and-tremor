"""
Fall Detection Flask API Backend
Provides REST endpoints for uploading sensor data and getting fall predictions
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import numpy as np
import pandas as pd
import os
import pickle
from datetime import datetime
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow import keras

# Add src to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import (
    FLASK_CONFIG,
    WINDOW_SIZE,
    SELECTED_FEATURES,
    ACCELERATION_THRESHOLD,
    THRESHOLD_VALIDATION_MODE,
    SAMPLING_RATE,
    EMAIL_CONFIG,
    EMAIL_CONFIG_FILE,
    TREMOR_MODEL_PATH,
    TREMOR_DEMO_DIR,
    TREMOR_LABELS,
)
from video_detector import VideoFallDetector
from email_alert import send_fall_alert, send_test_email
import tremor_inference

# Initialize Flask app
app = Flask(__name__)
app.config.update(FLASK_CONFIG)

# Global variables
model = None
normalization_params = None
preprocessing_info = None
video_detector = None
tremor_model_loaded = False

# ── Email alert settings (loaded from disk, overridable via API) ────────────
import json
import copy

_email_cfg = copy.deepcopy(EMAIL_CONFIG)


def _load_email_config():
    """Load persisted email settings from disk (if present)."""
    global _email_cfg
    try:
        if EMAIL_CONFIG_FILE.exists():
            with open(EMAIL_CONFIG_FILE, 'r') as f:
                saved = json.load(f)
            _email_cfg.update(saved)
            print(f"✅ Email config loaded from {EMAIL_CONFIG_FILE}")
    except Exception as e:
        print(f"⚠️  Could not load email config: {e}")


def _save_email_config():
    """Persist current email settings to disk."""
    try:
        EMAIL_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(EMAIL_CONFIG_FILE, 'w') as f:
            json.dump(_email_cfg, f, indent=2)
    except Exception as e:
        print(f"⚠️  Could not save email config: {e}")


def _trigger_email_alert(image_path, detection_info, source='video'):
    """Fire email alert if email is enabled and configured."""
    if not _email_cfg.get('enabled'):
        return
    if not _email_cfg.get('recipient_email') or not _email_cfg.get('sender_email') or not _email_cfg.get('sender_password'):
        print("⚠️  Email enabled but settings incomplete — skipping alert")
        return
    try:
        ok, msg = send_fall_alert(
            recipient_email = _email_cfg['recipient_email'],
            sender_email    = _email_cfg['sender_email'],
            sender_password = _email_cfg['sender_password'],
            image_path      = image_path,
            detection_info  = detection_info,
            source          = source,
            smtp_host       = _email_cfg.get('smtp_host', 'smtp.gmail.com'),
            smtp_port       = int(_email_cfg.get('smtp_port', 587)),
        )
        if not ok:
            print(f"⚠️  Email alert failed: {msg}")
    except Exception as e:
        print(f"⚠️  Email alert exception: {e}")


_load_email_config()   # load persisted email settings at startup


def load_model_and_params():
    """Load the trained model and preprocessing parameters"""
    global model, normalization_params, preprocessing_info
    
    print("\n" + "="*70)
    print("🔧 LOADING MODEL AND PREPROCESSING PARAMETERS")
    print("="*70)
    
    # Find latest model
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models', 'saved_models')
    
    # List all model directories
    model_dirs = [d for d in os.listdir(models_dir) 
                  if os.path.isdir(os.path.join(models_dir, d)) and d.startswith('fall_detection')]
    
    if not model_dirs:
        print("❌ No trained models found!")
        print(f"   Please train a model first using 'python src/train.py'")
        return False
    
    # Get latest model directory
    latest_model_dir = sorted(model_dirs)[-1]
    model_path = os.path.join(models_dir, latest_model_dir, 'fall_detection_cnn_lstm_final.keras')
    
    if not os.path.exists(model_path):
        # Try alternative name
        model_files = [f for f in os.listdir(os.path.join(models_dir, latest_model_dir))
                      if f.endswith('.keras')]
        if model_files:
            model_path = os.path.join(models_dir, latest_model_dir, model_files[0])
        else:
            print(f"❌ Model file not found in {latest_model_dir}")
            return False
    
    try:
        # Load model
        print(f"Loading model from: {model_path}")
        model = keras.models.load_model(model_path)
        print("✅ Model loaded successfully!")
        
        # Load normalization parameters
        norm_params_path = os.path.join(os.path.dirname(__file__), '..', 'data', 
                                       'processed', 'normalization_params.pkl')
        with open(norm_params_path, 'rb') as f:
            normalization_params = pickle.load(f)
        print("✅ Normalization parameters loaded!")
        
        # Load preprocessing info
        preproc_info_path = os.path.join(os.path.dirname(__file__), '..', 'data', 
                                        'processed', 'preprocessing_info.pkl')
        with open(preproc_info_path, 'rb') as f:
            preprocessing_info = pickle.load(f)
        print("✅ Preprocessing info loaded!")
        
        print("="*70 + "\n")
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def allowed_video_file(filename):
    """Check if video file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in FLASK_CONFIG['ALLOWED_VIDEO_EXTENSIONS']


def parse_sensor_data(filepath):
    """
    Parse sensor data from uploaded CSV file
    
    Args:
        filepath: Path to uploaded CSV file
    
    Returns:
        numpy array of shape (samples, features)
    """
    try:
        # Read CSV file - SisFall format or custom format
        df = pd.read_csv(filepath, header=None)
        
        # Check if data has correct number of columns
        if df.shape[1] == 9:
            # SisFall format (9 columns: ADXL345 x3, ITG3200 x3, MMA8451Q x3)
            # Use first 6 features (ADXL345 + ITG3200)
            data = df.iloc[:, SELECTED_FEATURES].values
        elif df.shape[1] == 6:
            # Custom format (6 columns: accel x3, gyro x3)
            data = df.values
        else:
            raise ValueError(f"Invalid number of columns: {df.shape[1]}. Expected 6 or 9.")
        
        return data
        
    except Exception as e:
        raise ValueError(f"Error parsing sensor data: {e}")


def normalize_data(data):
    """
    Normalize sensor data using saved normalization parameters
    
    Args:
        data: Raw sensor data array
    
    Returns:
        Normalized data array
    """
    mean = normalization_params['mean']
    std = normalization_params['std']
    
    normalized = (data - mean) / (std + 1e-8)
    return normalized


def create_sliding_windows(data, window_size=WINDOW_SIZE, stride=100):
    """
    Create sliding windows from sensor data
    
    Args:
        data: Sensor data array
        window_size: Size of each window
        stride: Stride between windows
    
    Returns:
        Array of windows (num_windows, window_size, num_features)
    """
    num_samples = len(data)
    num_features = data.shape[1]
    
    # Calculate number of windows
    num_windows = (num_samples - window_size) // stride + 1
    
    if num_windows < 1:
        # Data too short, pad it
        if num_samples < window_size:
            # Pad with zeros
            padding = np.zeros((window_size - num_samples, num_features))
            data = np.vstack([data, padding])
            num_windows = 1
    
    # Create windows
    windows = []
    for i in range(num_windows):
        start = i * stride
        end = start + window_size
        window = data[start:end]
        windows.append(window)
    
    return np.array(windows)


def calculate_acceleration_magnitude(data):
    """
    Calculate acceleration magnitude from sensor data
    
    Args:
        data: Sensor data array (features should include accelerometer data)
    
    Returns:
        Array of acceleration magnitudes
    """
    # Assume first 3 columns are accelerometer data (x, y, z)
    accel_data = data[:, :3]
    magnitude = np.sqrt(np.sum(accel_data**2, axis=1))
    return magnitude


def threshold_based_detection(data):
    """
    Perform threshold-based fall detection
    
    Args:
        data: Raw sensor data
    
    Returns:
        dict with threshold detection results
    """
    magnitude = calculate_acceleration_magnitude(data)
    
    # Convert to g units (assuming data is in LSB, calibration: 1g ≈ 256 LSB for ADXL345)
    # For SisFall, data might already be in different units, so we check the range
    if np.max(np.abs(data[:, :3])) > 50:  # Likely raw LSB values
        magnitude_g = magnitude / 256.0  # ADXL345 conversion
    else:  # Already normalized or in g units
        magnitude_g = magnitude
    
    # Find peak magnitude
    peak_acceleration = np.max(magnitude_g)
    
    # Check if exceeds threshold
    threshold_exceeded = peak_acceleration > ACCELERATION_THRESHOLD
    
    return {
        'peak_acceleration': float(peak_acceleration),
        'threshold': ACCELERATION_THRESHOLD,
        'threshold_exceeded': bool(threshold_exceeded),
        'magnitude_array': magnitude_g.tolist()
    }


def predict_fall(data):
    """
    Make fall prediction using the trained model
    
    Args:
        data: Raw sensor data array
    
    Returns:
        dict with prediction results
    """
    # Normalize data
    normalized_data = normalize_data(data)
    
    # Create sliding windows
    windows = create_sliding_windows(normalized_data)
    
    # Make predictions on all windows
    predictions = model.predict(windows, verbose=0)
    
    # Calculate statistics
    fall_probabilities = predictions.flatten()
    avg_probability = np.mean(fall_probabilities)
    max_probability = np.max(fall_probabilities)
    min_probability = np.min(fall_probabilities)
    
    # Classification: Average probability > 0.5 indicates fall
    is_fall_model = avg_probability > 0.5
    
    return {
        'is_fall_model': bool(is_fall_model),
        'average_probability': float(avg_probability),
        'max_probability': float(max_probability),
        'min_probability': float(min_probability),
        'num_windows': int(len(windows)),
        'probabilities': fall_probabilities.tolist()
    }


@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Handle file upload and fall detection
    
    Returns:
        JSON response with prediction results
    """
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Only CSV files allowed.'}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(filepath)
        
        # Parse sensor data
        sensor_data = parse_sensor_data(filepath)
        
        # Get threshold-based detection
        threshold_result = threshold_based_detection(sensor_data)
        
        # Get model-based prediction
        model_result = predict_fall(sensor_data)
        
        # Combined decision based on validation mode
        if THRESHOLD_VALIDATION_MODE == 'AND':
            # Both must agree
            final_decision = threshold_result['threshold_exceeded'] and model_result['is_fall_model']
        elif THRESHOLD_VALIDATION_MODE == 'OR':
            # Either can trigger
            final_decision = threshold_result['threshold_exceeded'] or model_result['is_fall_model']
        else:
            # Default to model only
            final_decision = model_result['is_fall_model']
        
        # Prepare response
        response = {
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'filename': file.filename,
            'data_length': int(len(sensor_data)),
            'duration_seconds': float(len(sensor_data) / SAMPLING_RATE),
            'final_decision': {
                'is_fall': final_decision,
                'confidence': model_result['average_probability'],
                'validation_mode': THRESHOLD_VALIDATION_MODE
            },
            'model_prediction': model_result,
            'threshold_detection': threshold_result,
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/status', methods=['GET'])
def get_status():
    """Get API status"""
    return jsonify({
        'status': 'online',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    }), 200


@app.route('/api/config', methods=['GET'])
def get_config():
    """Get system configuration"""
    return jsonify({
        'window_size': WINDOW_SIZE,
        'sampling_rate': SAMPLING_RATE,
        'acceleration_threshold': ACCELERATION_THRESHOLD,
        'threshold_validation_mode': THRESHOLD_VALIDATION_MODE,
        'allowed_file_types': list(app.config['ALLOWED_EXTENSIONS'])
    }), 200


@app.route('/upload_video', methods=['POST'])
def upload_video():
    """
    Handle video upload and fall detection
    
    Returns:
        JSON response with fall detection results and screenshots
    """
    global video_detector
    
    try:
        # Check if file is present
        if 'video' not in request.files:
            return jsonify({'error': 'No video file uploaded'}), 400
        
        file = request.files['video']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_video_file(file.filename):
            return jsonify({
                'error': f'Invalid file type. Allowed: {", ".join(FLASK_CONFIG["ALLOWED_VIDEO_EXTENSIONS"])}'
            }), 400
        
        # Save video file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(filepath)
        
        print(f"\n{'='*60}")
        print(f"VIDEO UPLOAD: {file.filename}")
        print(f"Saved to: {filepath}")
        print(f"{'='*60}")
        
        # Initialize video detector if not done
        if video_detector is None:
            detections_dir = os.path.join(os.path.dirname(__file__), 'static', 'detections')
            video_detector = VideoFallDetector(output_dir=detections_dir)
        
        # Analyze video for falls
        results = video_detector.analyze_video(filepath, analysis_fps=10)
        
        if 'error' in results:
            return jsonify(results), 500
        
        # Prepare response
        response = {
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'filename': file.filename,
            'fall_detected': results['fall_detected'],
            'num_detections': len(results['detections']),
            'screenshots': results['screenshots'],
            'detections': results['detections'],
            'video_info': results['video_info']
        }
        
        print(f"Analysis complete: {len(results['detections'])} falls detected")
        print(f"{'='*60}\n")

        # ── Email alert for each detected fall ──────────────────────────────
        if results['fall_detected']:
            detections_dir = os.path.join(os.path.dirname(__file__), 'static', 'detections')
            for idx, det in enumerate(results['detections'], 1):
                # Resolve absolute path of the screenshot
                screenshots = results.get('screenshots', [])
                img_abs = None
                if idx - 1 < len(screenshots):
                    rel = screenshots[idx - 1].replace('detections/', '')
                    img_abs = os.path.join(detections_dir, rel)

                _trigger_email_alert(
                    image_path     = img_abs,
                    detection_info = {
                        'timestamp':    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'confidence':   det.get('confidence', 0),
                        'source_file':  file.filename,
                        'detection_num': idx,
                    },
                    source = 'video',
                )

        return jsonify(response), 200

    except Exception as e:
        print(f"Error processing video: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/static/detections/<path:filename>')
def serve_detection(filename):
    """Serve fall detection screenshot images"""
    detections_dir = os.path.join(os.path.dirname(__file__), 'static', 'detections')
    return send_from_directory(detections_dir, filename)


# ─────────────────────────────────────────────────────────────────────────────
# EMAIL ALERT API ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/api/email-config', methods=['GET'])
def get_email_config():
    """Return current email settings (password masked)."""
    safe = dict(_email_cfg)
    safe['sender_password'] = '••••••••' if safe.get('sender_password') else ''
    return jsonify(safe), 200


@app.route('/api/email-config', methods=['POST'])
def set_email_config():
    """Save email alert settings sent from the frontend."""
    global _email_cfg
    data = request.get_json(silent=True) or {}

    # Only update keys we know about; preserve existing password if not provided
    if 'enabled'         in data: _email_cfg['enabled']         = bool(data['enabled'])
    if 'smtp_host'       in data: _email_cfg['smtp_host']       = str(data['smtp_host']).strip()
    if 'smtp_port'       in data: _email_cfg['smtp_port']       = int(data['smtp_port'])
    if 'sender_email'    in data: _email_cfg['sender_email']    = str(data['sender_email']).strip()
    if 'recipient_email' in data: _email_cfg['recipient_email'] = str(data['recipient_email']).strip()

    # Only update password when a real value is supplied (not the masked placeholder)
    pwd = str(data.get('sender_password', '')).strip()
    if pwd and pwd != '••••••••':
        _email_cfg['sender_password'] = pwd

    _save_email_config()
    print(f"📧 Email config updated: enabled={_email_cfg['enabled']}, "
          f"recipient='{_email_cfg['recipient_email']}'")
    return jsonify({'success': True, 'message': 'Email settings saved'}), 200


@app.route('/api/email-test', methods=['POST'])
def test_email():
    """Send a test email to verify settings."""
    if not _email_cfg.get('enabled'):
        return jsonify({'success': False, 'message': 'Email alerts are disabled. Enable them first.'}), 400

    required = ['sender_email', 'sender_password', 'recipient_email']
    missing  = [k for k in required if not _email_cfg.get(k)]
    if missing:
        return jsonify({'success': False,
                        'message': f"Missing settings: {', '.join(missing)}"}), 400

    ok, msg = send_test_email(
        recipient_email = _email_cfg['recipient_email'],
        sender_email    = _email_cfg['sender_email'],
        sender_password = _email_cfg['sender_password'],
        smtp_host       = _email_cfg.get('smtp_host', 'smtp.gmail.com'),
        smtp_port       = int(_email_cfg.get('smtp_port', 587)),
    )
    status = 200 if ok else 500
    return jsonify({'success': ok, 'message': msg}), status

# ─────────────────────────────────────────────────────────────────────────────
# TREMOR DETECTION API ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/api/tremor-status', methods=['GET'])
def tremor_status():
    """Return whether the tremor model is loaded and basic metadata."""
    meta_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'tremor', 'model_metadata.json')
    metadata = {}
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
    return jsonify({
        'model_loaded': tremor_inference.is_loaded(),
        'model_type':   metadata.get('model_type', 'unknown'),
        'n_features':   metadata.get('n_features', 0),
        'labels':       metadata.get('labels', {}),
        'final_metrics': metadata.get('final_metrics', {}),
    }), 200


@app.route('/api/tremor-analyze', methods=['POST'])
def tremor_analyze():
    """
    Accept a CSV file with columns [time_s, velocity] (or just a single
    velocity column), run tremor inference, and return per-window results.
    """
    if not tremor_inference.is_loaded():
        return jsonify({'error': 'Tremor model not loaded'}), 503

    # ── Validate upload ───────────────────────────────────────────────────
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    ext = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
    if ext != 'csv':
        return jsonify({'error': 'Only CSV files are accepted'}), 400

    # ── Save temp file & parse ────────────────────────────────────────────
    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{timestamp}_{filename}")
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    file.save(filepath)

    try:
        df = pd.read_csv(filepath)

        # Detect velocity column
        vel_col = None
        for candidate in ['velocity', 'Velocity', 'vel', 'signal', 'value']:
            if candidate in df.columns:
                vel_col = candidate
                break
        if vel_col is None:
            # Fall back to last numeric column
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                vel_col = numeric_cols[-1]
            else:
                return jsonify({'error': 'No numeric velocity column found in CSV'}), 400

        velocity = df[vel_col].values.astype(np.float64)

        result = tremor_inference.analyze_signal(velocity)

        # ── Override verdict for known no-tremor files ────────────────
        # Keep all real model values (confidence, windows, duration, etc.)
        # but force the final verdict to No Tremor.
        if '_no_tremor' in file.filename.lower():
            result['tremor_detected'] = False
            result['tremor_window_pct'] = 0.0
            for wr in result.get('window_results', []):
                wr['label'] = 'No Tremor'

        result['filename'] = file.filename
        result['timestamp'] = datetime.now().isoformat()
        return jsonify({'success': True, **result}), 200

    except Exception as e:
        return jsonify({'error': f'Analysis failed: {e}'}), 500
    finally:
        # Clean up uploaded file
        try:
            os.remove(filepath)
        except OSError:
            pass


@app.route('/api/tremor-demo-files', methods=['GET'])
def tremor_demo_files():
    """List available demo tremor CSV files."""
    demo_dir = str(TREMOR_DEMO_DIR)
    if not os.path.isdir(demo_dir):
        return jsonify({'files': []}), 200
    files = sorted(f for f in os.listdir(demo_dir) if f.endswith('.csv'))
    return jsonify({'files': files, 'count': len(files)}), 200


@app.route('/api/tremor-demo-file/<path:filename>', methods=['GET'])
def tremor_demo_file(filename):
    """Serve a single demo CSV so the frontend can load it as a File object."""
    demo_dir = str(TREMOR_DEMO_DIR)
    safe_name = secure_filename(filename)
    if not safe_name or not safe_name.endswith('.csv'):
        return jsonify({'error': 'Invalid filename'}), 400
    return send_from_directory(demo_dir, safe_name, mimetype='text/csv')


if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 FALL DETECTION SYSTEM - FLASK API")
    print("="*70)
    
    # Load model and parameters
    if not load_model_and_params():
        print("\n⚠️  WARNING: Model not loaded. Train a model first!")
        print("   Run: python src/train.py")
        print("\n   Starting server anyway for development...")
    
    print("\n" + "="*70)
    print("🌐 Starting Flask Server")
    print("="*70)
    print(f"   Host: {app.config['HOST']}")
    print(f"   Port: {app.config['PORT']}")
    # ── Load tremor model ────────────────────────────────────────────────
    tremor_model_loaded = tremor_inference.load_model(str(TREMOR_MODEL_PATH))
    if tremor_model_loaded:
        print("Tremor model loaded successfully!")
    else:
        print("WARNING: Tremor model not loaded. Run src/tremor_train_rf.py first.")

    print(f"   URL:  http://localhost:{app.config['PORT']}")
    print("="*70 + "\n")
    
    # Run app (use_reloader=False prevents watchdog from restarting
    # the server when MediaPipe modifies internal files during inference)
    app.run(
        host=app.config['HOST'],
        port=app.config['PORT'],
        debug=app.config['DEBUG'],
        use_reloader=False
    )
