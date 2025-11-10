import os

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

# Camera settings
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
FPS = 30

# MediaPipe settings
MAX_NUM_HANDS = 2
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.5

# Model settings
INPUT_SHAPE = (21, 3)  # 21 hand landmarks, 3 coordinates each
NUM_CLASSES = 26  # A-Z alphabet signs
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001

# Data collection settings
SAMPLES_PER_SIGN = 25
SIGN_DELAY = 2  
COOLDOWN = 1  

# Supported signs (American Sign Language alphabet)
SIGNS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
]

# Colors for visualization
COLORS = {
    'hand_landmarks': (0, 255, 0),  # Green
    'hand_connections': (255, 0, 0),  # Red
    'text': (255, 255, 255),  # White
    'background': (0, 0, 0)  # Black
}
