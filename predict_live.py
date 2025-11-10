import cv2
import numpy as np
import os
import time
from collections import deque
from utils.config import *
from utils.hand_utils import HandDetector
from train_model import SignLanguageModel

class LiveSignPredictor:
    def __init__(self, model_path: str = None):
        """
        Initialize the live sign language predictor.
        
        Args:
            model_path: Path to the trained model file
        """
        self.hand_detector = HandDetector()
        self.model = SignLanguageModel()
        self.cap = None
        
        # Prediction smoothing
        self.prediction_history = deque(maxlen=5)
        self.confidence_threshold = 0.7
        self.stable_predictions = 3  # Number of consistent predictions needed
        
        # Load model
        self._load_model(model_path)
        
        # Create models directory if it doesn't exist
        os.makedirs(MODELS_DIR, exist_ok=True)
    
    def _load_model(self, model_path: str = None):
        """Load the trained model."""
        if model_path is None:
            # Look for the most recent model file
            model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.h5')]
            if not model_files:
                raise FileNotFoundError("No trained model found. Train a model first.")
            
            model_path = os.path.join(MODELS_DIR, model_files[-1])
        
        try:
            self.model.load_model(model_path)
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def start_camera(self):
        """Initialize and start the webcam."""
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, FPS)
        
        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera")
        
        print("Camera started successfully")
    
    def stop_camera(self):
        """Stop and release the webcam."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Camera stopped")
    
    def predict_live(self):
        """Run live sign language prediction."""
        print("Starting live sign language prediction...")
        print("Press 'q' to quit")
        
        try:
            self.start_camera()
            
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    continue
                
                # Process frame for hand detection
                processed_frame, landmarks_list = self.hand_detector.detect_hands(frame)
                
                # Handle prediction if hands are detected
                if landmarks_list:
                    self._handle_prediction(processed_frame, landmarks_list)
                else:
                    # No hands detected
                    self._draw_no_hands_message(processed_frame)
                
                # Display frame
                cv2.imshow('Live Sign Language Recognition', processed_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
        
        finally:
            self.stop_camera()
            self.hand_detector.cleanup()
    
    def _handle_prediction(self, frame, landmarks_list):
        """Handle sign prediction for detected hands."""
        if len(landmarks_list) > 0:
            # Use the first detected hand
            landmarks = landmarks_list[0]
            
            # Extract landmark coordinates
            coords = self.hand_detector.extract_landmarks(landmarks)
            
            # Make prediction
            try:
                predicted_sign, confidence = self.model.predict_sign(coords)
                
                # Add to prediction history
                self.prediction_history.append((predicted_sign, confidence))
                
                # Get stable prediction
                stable_prediction = self._get_stable_prediction()
                
                # Draw results
                self._draw_prediction_results(frame, landmarks, stable_prediction, confidence)
                
            except Exception as e:
                print(f"Prediction error: {e}")
                self._draw_error_message(frame, landmarks)
    
    def _get_stable_prediction(self) -> tuple:
        """
        Get a stable prediction from recent history.
        
        Returns:
            Tuple of (predicted_sign, confidence) or (None, 0) if not stable
        """
        if len(self.prediction_history) < self.stable_predictions:
            return None, 0
        
        # Get recent predictions
        recent_predictions = [pred[0] for pred in list(self.prediction_history)[-self.stable_predictions:]]
        recent_confidences = [pred[1] for pred in list(self.prediction_history)[-self.stable_predictions:]]
        
        # Check if predictions are consistent
        if len(set(recent_predictions)) == 1:  # All predictions are the same
            avg_confidence = np.mean(recent_confidences)
            if avg_confidence >= self.confidence_threshold:
                return recent_predictions[0], avg_confidence
        
        return None, 0
    
    def _draw_prediction_results(self, frame, landmarks, prediction, confidence):
        """Draw prediction results on the frame."""
        # Get hand bounding box
        x_min, y_min, x_max, y_max = self.hand_detector.get_hand_bbox(landmarks, frame.shape)
        
        if prediction and confidence >= self.confidence_threshold:
            # High confidence prediction - draw green box
            color = (0, 255, 0)
            box_text = f"{prediction} ({confidence:.2f})"
        else:
            # Low confidence or no stable prediction - draw yellow box
            color = (0, 255, 255)
            box_text = "Processing..."
        
        # Draw bounding box
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
        
        # Draw prediction text
        cv2.putText(frame, box_text, (x_min, y_min - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw confidence bar
        if prediction:
            self._draw_confidence_bar(frame, confidence, x_min, y_max + 20)
    
    def _draw_confidence_bar(self, frame, confidence, x, y):
        """Draw a confidence bar."""
        bar_width = 100
        bar_height = 10
        
        # Background bar (gray)
        cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height), (128, 128, 128), -1)
        
        # Confidence bar (green to red based on confidence)
        confidence_width = int(bar_width * confidence)
        if confidence > 0.8:
            color = (0, 255, 0)  # Green
        elif confidence > 0.6:
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 0, 255)  # Red
        
        cv2.rectangle(frame, (x, y), (x + confidence_width, y + bar_height), color, -1)
        
        # Border
        cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height), (255, 255, 255), 1)
    
    def _draw_no_hands_message(self, frame):
        """Draw message when no hands are detected."""
        h, w = frame.shape[:2]
        
        # Draw background rectangle
        cv2.rectangle(frame, (w//2 - 200, h//2 - 50), (w//2 + 200, h//2 + 50), (0, 0, 0), -1)
        
        # Draw text
        cv2.putText(frame, "No hands detected", (w//2 - 150, h//2 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, "Show your hand to the camera", (w//2 - 180, h//2 + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    def _draw_error_message(self, frame, landmarks):
        """Draw error message when prediction fails."""
        x_min, y_min, x_max, y_max = self.hand_detector.get_hand_bbox(landmarks, frame.shape)
        
        # Draw red box
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
        
        # Draw error text
        cv2.putText(frame, "Prediction Error", (x_min, y_min - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    def _draw_ui(self, frame):
        """Draw UI elements on the frame."""
        h, w = frame.shape[:2]
        
        # Draw title
        cv2.putText(frame, "Live Sign Language Recognition", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Draw instructions
        cv2.putText(frame, "Show your hand to the camera", (10, h - 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "Press 'q' to quit", (10, h - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw prediction history
        self._draw_prediction_history(frame)
    
    def _draw_prediction_history(self, frame):
        """Draw recent prediction history."""
        if not self.prediction_history:
            return
        
        # Get recent predictions
        recent = list(self.prediction_history)[-3:]  # Last 3 predictions
        
        y_offset = 80
        for i, (sign, conf) in enumerate(recent):
            text = f"{sign}: {conf:.2f}"
            color = (0, 255, 0) if conf >= self.confidence_threshold else (0, 255, 255)
            cv2.putText(frame, text, (10, y_offset + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def main():
    """Main function to run live sign language prediction."""
    print("=== Live Sign Language Recognition ===")
    
    try:
        # Initialize predictor
        predictor = LiveSignPredictor()
        
        # Start live prediction
        predictor.predict_live()
        
    except KeyboardInterrupt:
        print("\nPrediction interrupted by user")
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
