import cv2
import numpy as np
from typing import List, Tuple, Optional
from .config import *

# Try to import MediaPipe with error handling
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    print("MediaPipe imported successfully!")
except ImportError as e:
    print(f"MediaPipe import failed: {e}")
    print("Please install Visual C++ Redistributable and restart your computer.")
    print("Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe")
    MEDIAPIPE_AVAILABLE = False
    mp = None

class HandDetector:
    def __init__(self):
        """Initialize MediaPipe hands module with configuration."""
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe is not available. Please install Visual C++ Redistributable and restart your computer.")
        
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            max_num_hands=MAX_NUM_HANDS,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE
        )
    
    def detect_hands(self, image: np.ndarray) -> Tuple[np.ndarray, List]:
        """
        Detect hands in the image and return processed image with landmarks.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Tuple of (processed_image, landmarks_list)
        """
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = self.hands.process(rgb_image)
        
        # Convert back to BGR for OpenCV
        processed_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        
        landmarks_list = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks_list.append(hand_landmarks)
        
        return processed_image, landmarks_list
    
    def extract_landmarks(self, landmarks) -> np.ndarray:

        coords = []
        for landmark in landmarks.landmark:
            coords.append([landmark.x, landmark.y, landmark.z])
        return np.array(coords)
    
    def draw_hands(self, image: np.ndarray, landmarks_list: List) -> np.ndarray:
        
        for landmarks in landmarks_list:
            # Draw landmarks
            self.mp_drawing.draw_landmarks(
                image,
                landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
        return image
    
    def get_hand_bbox(self, landmarks, 
                      image_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """
        Get bounding box coordinates for a hand.
        
        Args:
            landmarks: MediaPipe hand landmarks object
            image_shape: Tuple of (height, width)
            
        Returns:
            Tuple of (x_min, y_min, x_max, y_max)
        """
        h, w = image_shape[:2]
        x_coords = [landmark.x * w for landmark in landmarks.landmark]
        y_coords = [landmark.y * h for landmark in  landmarks.landmark]
        
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))
        
        return x_min, y_min, x_max, y_max
    
    def is_hand_centered(self, landmarks, 
                         image_shape: Tuple[int, int], 
                         center_threshold: float = 0.3) -> bool:
        """
        Check if hand is centered in the image.
        
        Args:
            landmarks: MediaPipe hand landmarks object
            image_shape: Tuple of (height, width)
            center_threshold: Threshold for considering hand centered
            
        Returns:
            True if hand is centered, False otherwise
        """
        h, w = image_shape[:2]
        
        # Get wrist position (landmark 0)
        wrist = landmarks.landmark[0]
        wrist_x, wrist_y = wrist.x, wrist.y
        
        # Check if wrist is in center region
        x_centered = abs(wrist_x - 0.5) < center_threshold
        y_centered = abs(wrist_y - 0.5) < center_threshold
        
        return x_centered and y_centered
    
    def cleanup(self):
        """Clean up MediaPipe resources."""
        self.hands.close()

def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    Normalize landmarks to be translation and scale invariant.
    
    Args:
        landmarks: Landmark coordinates of shape (21, 3)
        
    Returns:
        Normalized landmarks
    """
    # Center landmarks around wrist (landmark 0)
    wrist = landmarks[0]
    centered = landmarks - wrist
    
    # Scale by distance from wrist to middle finger tip (landmark 12)
    scale_factor = np.linalg.norm(centered[12])
    if scale_factor > 0:
        normalized = centered / scale_factor
    else:
        normalized = centered
    
    return normalized

def augment_landmarks(landmarks: np.ndarray, 
                     noise_factor: float = 0.01,
                     rotation_angle: float = 0.0) -> np.ndarray:
    """
    Apply data augmentation to landmarks.
    
    Args:
        landmarks: Input landmarks
        noise_factor: Standard deviation of Gaussian noise
        rotation_angle: Rotation angle in radians
        
    Returns:
        Augmented landmarks
    """
    # Add Gaussian noise
    noise = np.random.normal(0, noise_factor, landmarks.shape)
    augmented = landmarks + noise
    
    # Apply rotation around z-axis (if needed)
    if rotation_angle != 0:
        cos_a, sin_a = np.cos(rotation_angle), np.sin(rotation_angle)
        rotation_matrix = np.array([[cos_a, -sin_a, 0],
                                   [sin_a, cos_a, 0],
                                   [0, 0, 1]])
        augmented = np.dot(augmented, rotation_matrix.T)
    
    return augmented
