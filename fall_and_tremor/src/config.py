"""
Configuration file for Fall Detection System
Contains all hyperparameters, paths, and settings
"""

import os
from pathlib import Path

# =====================================================
# PROJECT PATHS
# =====================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
SRC_DIR = BASE_DIR / "src"
APP_DIR = BASE_DIR / "app"

# Data subdirectories
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"

# Model subdirectories
SAVED_MODELS_DIR = MODEL_DIR / "saved_models"
CHECKPOINTS_DIR = MODEL_DIR / "checkpoints"
LOGS_DIR = MODEL_DIR / "logs"

# SisFall Dataset Path (update this to your actual path)
SISFALL_DATASET_PATH = BASE_DIR / "SisFall_dataset"

# =====================================================
# TREMOR DETECTION PATHS & CONFIG
# =====================================================
TREMOR_DATASET_DIR  = BASE_DIR / "effect-of-deep-brain-stimulation-on-parkinsonian-tremor-1.0.0"
TREMOR_DATA_DIR     = DATA_DIR / "tremor"
TREMOR_DEMO_DIR     = BASE_DIR / "test_data" / "tremor_samples"
TREMOR_MODEL_PATH   = MODEL_DIR / "tremor_model.joblib"   # XGBoost pipeline

# Windowing
TREMOR_SAMPLE_RATE  = 100      # Hz (all recordings in this dataset)
TREMOR_WINDOW_SEC   = 2        # seconds per window
TREMOR_STEP_SEC     = 1        # step (50% overlap)
TREMOR_WINDOW_SIZE  = TREMOR_SAMPLE_RATE * TREMOR_WINDOW_SEC   # 200 samples
TREMOR_STEP_SIZE    = TREMOR_SAMPLE_RATE * TREMOR_STEP_SEC     # 100 samples

# Labels
TREMOR_LABELS       = {0: "No Tremor", 1: "Tremor Detected"}
TREMOR_LABEL_TREMOR    = 1
TREMOR_LABEL_NO_TREMOR = 0

# =====================================================
# DATASET CONFIGURATION
# =====================================================
# SisFall Activities
FALL_ACTIVITIES = [
    'F01', 'F02', 'F03', 'F04', 'F05', 
    'F06', 'F07', 'F08', 'F09', 'F10',
    'F11', 'F12', 'F13', 'F14', 'F15'
]

ADL_ACTIVITIES = [
    'D01', 'D02', 'D03', 'D04', 'D05',
    'D06', 'D07', 'D08', 'D09', 'D10',
    'D11', 'D12', 'D13', 'D14', 'D15',
    'D16', 'D17', 'D18', 'D19'
]

# Activity descriptions
ACTIVITY_DESCRIPTIONS = {
    # Falls
    'F01': 'Fall forward while walking caused by a slip',
    'F02': 'Fall backward while walking caused by a slip',
    'F03': 'Lateral fall while walking caused by a slip',
    'F04': 'Fall forward while walking caused by a trip',
    'F05': 'Fall forward while jogging caused by a trip',
    'F06': 'Vertical fall while walking caused by fainting',
    'F07': 'Fall while walking, with use of hands to dampen fall',
    'F08': 'Fall forward when trying to get up',
    'F09': 'Lateral fall when trying to get up',
    'F10': 'Fall forward when trying to sit down',
    'F11': 'Fall backward when trying to sit down',
    'F12': 'Lateral fall when trying to sit down',
    'F13': 'Fall forward while sitting, caused by fainting',
    'F14': 'Fall backward while sitting, caused by fainting',
    'F15': 'Lateral fall while sitting, caused by fainting',
    
    # ADLs (Activities of Daily Living)
    'D01': 'Walking slowly',
    'D02': 'Walking quickly',
    'D03': 'Jogging slowly',
    'D04': 'Jogging quickly',
    'D05': 'Walking upstairs and downstairs slowly',
    'D06': 'Walking upstairs and downstairs quickly',
    'D07': 'Slowly sit in half height chair and up slowly',
    'D08': 'Quickly sit in half height chair and up quickly',
    'D09': 'Slowly sit in low height chair and up slowly',
    'D10': 'Quickly sit in low height chair and up quickly',
    'D11': 'Trying to get up and collapse into chair',
    'D12': 'Sitting, lying slowly, and sit again',
    'D13': 'Sitting, lying quickly, and sit again',
    'D14': 'Being on back, change to lateral position',
    'D15': 'Standing, slowly bending at knees',
    'D16': 'Standing, slowly bending without bending knees',
    'D17': 'Standing, get into car, remain seated, get out',
    'D18': 'Stumble while walking',
    'D19': 'Gently jump without falling',
}

# Subject categories
ADULT_SUBJECTS = [f'SA{i:02d}' for i in range(1, 24)]  # SA01-SA23
ELDERLY_SUBJECTS = [f'SE{i:02d}' for i in range(1, 16)]  # SE01-SE15
ALL_SUBJECTS = ADULT_SUBJECTS + ELDERLY_SUBJECTS

# =====================================================
# DATA PREPROCESSING CONFIGURATION
# =====================================================
# Sampling rate
SAMPLING_RATE = 200  # Hz (SisFall dataset)

# Sliding window parameters
WINDOW_DURATION = 1.0  # seconds
WINDOW_SIZE = int(SAMPLING_RATE * WINDOW_DURATION)  # 200 samples
OVERLAP_RATIO = 0.5  # 50% overlap
OVERLAP = int(WINDOW_SIZE * OVERLAP_RATIO)  # 100 samples
STRIDE = WINDOW_SIZE - OVERLAP  # 100 samples

# Sensor configuration
# SisFall columns: [ADXL345_X, ADXL345_Y, ADXL345_Z, ITG3200_X, ITG3200_Y, ITG3200_Z, MMA8451Q_X, MMA8451Q_Y, MMA8451Q_Z]
SENSOR_COLUMNS = {
    'ADXL345': [0, 1, 2],      # Accelerometer 1 (columns 0-2)
    'ITG3200': [3, 4, 5],      # Gyroscope (columns 3-5)
    'MMA8451Q': [6, 7, 8],     # Accelerometer 2 (columns 6-8)
}

# Select which sensors to use
USE_SENSORS = ['ADXL345', 'ITG3200']  # Using accel + gyro (6 features)
# USE_SENSORS = ['ADXL345']  # Using only accelerometer (3 features)
# USE_SENSORS = ['ADXL345', 'ITG3200', 'MMA8451Q']  # Using all sensors (9 features)

# Get feature indices based on selected sensors
SELECTED_FEATURES = []
for sensor in USE_SENSORS:
    SELECTED_FEATURES.extend(SENSOR_COLUMNS[sensor])
NUM_FEATURES = len(SELECTED_FEATURES)

# Normalization method
NORMALIZATION_METHOD = 'standardization'  # 'standardization' or 'minmax'

# Class balancing
BALANCE_CLASSES = True
BALANCE_METHOD = 'undersample'  # 'undersample', 'oversample', or 'smote'

# Data split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_STATE = 42

# =====================================================
# MODEL CONFIGURATION
# =====================================================
# Model architecture selection
MODEL_ARCHITECTURE = 'CNN_LSTM'  # Options: 'CNN', 'LSTM', 'CNN_LSTM'

# Model hyperparameters
MODEL_CONFIG = {
    'CNN': {
        'conv1_filters': 64,
        'conv1_kernel': 5,
        'conv2_filters': 128,
        'conv2_kernel': 5,
        'conv3_filters': 256,
        'conv3_kernel': 3,
        'dense1_units': 128,
        'dense2_units': 64,
        'dropout_rate': 0.5,
        'dropout_rate2': 0.3,
    },
    'LSTM': {
        'lstm1_units': 128,
        'lstm2_units': 64,
        'dropout_rate': 0.3,
        'dense_units': 64,
    },
    'CNN_LSTM': {
        'conv1_filters': 64,
        'conv1_kernel': 5,
        'conv2_filters': 128,
        'conv2_kernel': 3,
        'lstm_units': 64,
        'dropout_rate': 0.4,
        'dense_units': 64,
        'dense_dropout': 0.3,
    }
}

# =====================================================
# TRAINING CONFIGURATION
# =====================================================
# Training hyperparameters
BATCH_SIZE = 64  # Optimal for RTX 3050
EPOCHS = 50
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.5  # Dropout for regularization
L2_REGULARIZATION = 0.001  # L2 weight regularization
OPTIMIZER = 'adam'  # 'adam', 'sgd', 'rmsprop'
LOSS_FUNCTION = 'binary_crossentropy'

# Metrics to track
METRICS = [
    'accuracy',
    'precision',
    'recall',
    'AUC',
]

# Callbacks configuration
EARLY_STOPPING_CONFIG = {
    'monitor': 'val_loss',
    'patience': 10,
    'restore_best_weights': True,
    'verbose': 1,
}

MODEL_CHECKPOINT_CONFIG = {
    'monitor': 'val_accuracy',
    'save_best_only': True,
    'save_weights_only': False,
    'verbose': 1,
}

REDUCE_LR_CONFIG = {
    'monitor': 'val_loss',
    'factor': 0.5,
    'patience': 5,
    'min_lr': 1e-6,
    'verbose': 1,
}

# Class weights (for imbalanced data)
USE_CLASS_WEIGHTS = True

# Data augmentation (optional)
USE_DATA_AUGMENTATION = False
AUGMENTATION_CONFIG = {
    'noise_level': 0.01,
    'time_shift_range': 10,
    'magnitude_warp': True,
}

# =====================================================
# THRESHOLD-BASED VALIDATION CONFIGURATION
# =====================================================
# Acceleration magnitude threshold (in g units)
ACCELERATION_THRESHOLD = 2.5  # g (gravity units)
# 1g = 9.81 m/s²
# Falls typically have peak acceleration > 2.5g

# Threshold validation mode
THRESHOLD_VALIDATION_MODE = 'AND'  # 'AND' or 'OR'
# 'AND': Both model and threshold must agree (fewer false positives)
# 'OR': Either model or threshold triggers (higher sensitivity)

# =====================================================
# FLASK APPLICATION CONFIGURATION
# =====================================================
FLASK_CONFIG = {
    'DEBUG': True,
    'HOST': '0.0.0.0',
    'PORT': 5000,
    'SECRET_KEY': 'fall-detection-secret-key-2026',
    'MAX_CONTENT_LENGTH': 100 * 1024 * 1024,  # 100MB max file size (for videos)
    'UPLOAD_FOLDER': APP_DIR / 'uploads',
    'ALLOWED_EXTENSIONS': {'csv', 'txt'},
    'ALLOWED_VIDEO_EXTENSIONS': {'mp4', 'avi', 'mov', 'mkv', 'webm'},
}

# API response configuration
API_RESPONSE_CONFIG = {
    'include_details': True,
    'include_timestamp': True,
    'include_activity_analysis': True,
}

# =====================================================
# LOGGING CONFIGURATION
# =====================================================
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': BASE_DIR / 'logs' / 'fall_detection.log',
}

# =====================================================
# GPU CONFIGURATION
# =====================================================
# Enable/disable GPU usage
USE_GPU = True

# Enable GPU memory growth (prevents TensorFlow from allocating all GPU memory)
ENABLE_GPU_MEMORY_GROWTH = True
GPU_MEMORY_GROWTH = True  # Alias for compatibility

# GPU device selection (-1 for CPU, 0 for first GPU, etc.)
GPU_DEVICE = 0

# Mixed precision training (faster on RTX 3050)
USE_MIXED_PRECISION = True

# =====================================================
# EVALUATION CONFIGURATION
# =====================================================
EVALUATION_CONFIG = {
    'save_confusion_matrix': True,
    'save_roc_curve': True,
    'save_precision_recall_curve': True,
    'save_classification_report': True,
    'generate_misclassification_analysis': True,
}

# =====================================================
# EMAIL ALERT CONFIGURATION
# =====================================================
# Default SMTP settings (Gmail). Change as needed.
# For Gmail: generate an App Password at myaccount.google.com/apppasswords
EMAIL_CONFIG = {
    'enabled':          False,               # master on/off switch
    'smtp_host':        'smtp.gmail.com',    # SMTP server
    'smtp_port':        587,                 # TLS port
    'sender_email':     'taryareddy123@gmail.com',                  # your email address
    'sender_password':  'pwey umjm cmsm gdzw',                  # App Password (not regular password)
    'recipient_email':  'akashreddy12390@gmail.com',                  # comma-separated recipients
}

# Path where email settings are persisted between restarts
EMAIL_CONFIG_FILE = BASE_DIR / 'app' / 'email_settings.json'

# =====================================================
# MODEL EXPORT CONFIGURATION
# =====================================================
EXPORT_FORMATS = ['h5', 'savedmodel', 'onnx']
MODEL_NAME = 'fall_detection_model'

# =====================================================
# UTILITY FUNCTIONS
# =====================================================
def get_model_path(model_name=None, format='h5'):
    """Get the path for saving/loading models"""
    name = model_name or MODEL_NAME
    return SAVED_MODELS_DIR / f"{name}.{format}"

def get_checkpoint_path(model_name=None):
    """Get the path for model checkpoints"""
    name = model_name or MODEL_NAME
    return CHECKPOINTS_DIR / f"{name}_checkpoint.h5"

def get_log_dir(model_name=None):
    """Get the TensorBoard log directory"""
    name = model_name or MODEL_NAME
    return LOGS_DIR / name

def create_directories():
    """Create all necessary directories"""
    directories = [
        DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, SPLITS_DIR,
        MODEL_DIR, SAVED_MODELS_DIR, CHECKPOINTS_DIR, LOGS_DIR,
        APP_DIR / 'uploads', BASE_DIR / 'logs'
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    print("✅ All directories created successfully")

def print_config():
    """Print configuration summary"""
    print("\n" + "="*60)
    print("FALL DETECTION SYSTEM - CONFIGURATION")
    print("="*60)
    print(f"Model Architecture: {MODEL_ARCHITECTURE}")
    print(f"Window Size: {WINDOW_SIZE} samples ({WINDOW_DURATION}s @ {SAMPLING_RATE}Hz)")
    print(f"Features: {NUM_FEATURES} ({', '.join(USE_SENSORS)})")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Acceleration Threshold: {ACCELERATION_THRESHOLD}g")
    print(f"GPU Memory Growth: {ENABLE_GPU_MEMORY_GROWTH}")
    print(f"Mixed Precision: {USE_MIXED_PRECISION}")
    print("="*60 + "\n")

if __name__ == "__main__":
    create_directories()
    print_config()
