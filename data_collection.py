import cv2
import numpy as np
import pandas as pd
import os
import time
from datetime import datetime
from utils.config import *
from utils.hand_utils import HandDetector

class SignDataCollector:
    def __init__(self):
        """Initialize the sign language data collector."""
        self.hand_detector = HandDetector()
        self.cap = None
        self.data = []
        self.current_sign = None
        self.sample_count = 0
        self.collection_active = False
        
        # Create data directory if it doesn't exist
        os.makedirs(DATA_DIR, exist_ok=True)
        
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
    
    def collect_sign_data(self, sign: str, num_samples: int = None):
        """
        Collect data for a specific sign.
        
        Args:
            sign: The sign to collect data for
            num_samples: Number of samples to collect (default from config)
        """
        if num_samples is None:
            num_samples = SAMPLES_PER_SIGN
        
        self.current_sign = sign
        self.sample_count = 0
        self.collection_active = True
        
        print(f"\nCollecting data for sign '{sign}'")
        print(f"Target samples: {num_samples}")
        print("Press 'q' to quit, 's' to skip current sign")
        
        while self.sample_count < num_samples and self.collection_active:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # Process frame
            processed_frame, landmarks_list = self.hand_detector.detect_hands(frame)
            
            # Draw UI elements
            self._draw_ui(processed_frame, sign, num_samples)
            
            # Handle hand detection and data collection
            if landmarks_list:
                self._handle_hand_detection(processed_frame, landmarks_list, sign)
            
            # Display frame
            cv2.imshow('Sign Language Data Collection', processed_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.collection_active = False
                break
            elif key == ord('s'):
                print(f"Skipping sign '{sign}'")
                break
        
        print(f"Collected {self.sample_count} samples for sign '{sign}'")
    
    def _handle_hand_detection(self, frame, landmarks_list, sign):
        """Handle hand detection and data collection logic."""
        if len(landmarks_list) > 0:
            # Use the first detected hand
            landmarks = landmarks_list[0]
            
            # Check if hand is centered
            if self.hand_detector.is_hand_centered(landmarks, frame.shape):
                # Draw green bounding box for centered hand
                x_min, y_min, x_max, y_max = self.hand_detector.get_hand_bbox(landmarks, frame.shape)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                
                # Add "Ready to capture" text
                cv2.putText(frame, "Ready to capture!", (x_min, y_min - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Auto-capture after delay
                if not hasattr(self, '_last_capture_time'):
                    self._last_capture_time = 0
                
                current_time = time.time()
                if current_time - self._last_capture_time > COOLDOWN:
                    self._capture_sample(landmarks, sign)
                    self._last_capture_time = current_time
            else:
                # Draw red bounding box for non-centered hand
                x_min, y_min, x_max, y_max = self.hand_detector.get_hand_bbox(landmarks, frame.shape)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                
                # Add "Center your hand" text
                cv2.putText(frame, "Center your hand", (x_min, y_min - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    def _capture_sample(self, landmarks, sign):
        """Capture a sample of hand landmarks."""
        # Extract landmark coordinates
        coords = self.hand_detector.extract_landmarks(landmarks)
        
        # Flatten coordinates to 1D array
        coords_flat = coords.flatten()
        
        # Add to data
        self.data.append({
            'sign': sign,
            'landmarks': coords_flat.tolist(),
            'timestamp': datetime.now().isoformat()
        })
        
        self.sample_count += 1
        print(f"Captured sample {self.sample_count}")
    
    def _draw_ui(self, frame, sign, target_samples):
        """Draw UI elements on the frame."""
        # Draw background rectangle for text
        cv2.rectangle(frame, (10, 10), (400, 120), (0, 0, 0), -1)
        
        # Draw current sign and progress
        cv2.putText(frame, f"Sign: {sign}", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.putText(frame, f"Progress: {self.sample_count}/{target_samples}", (20, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw instructions
        cv2.putText(frame, "Instructions:", (20, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "Center your hand in the frame", (20, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw controls
        cv2.putText(frame, "Controls: 'q'=quit, 's'=skip", (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def collect_all_signs(self):
        """Collect data for all supported signs."""
        print("Starting data collection for all signs...")
        print(f"Supported signs: {', '.join(SIGNS)}")
        
        try:
            self.start_camera()
            # Ensure collection runs; previously loop exited immediately when False
            self.collection_active = True
            
            for sign in SIGNS:
                if not self.collection_active:
                    break
                
                # Show sign instruction
                self._show_sign_instruction(sign)
                
                # Collect data for this sign
                self.collect_sign_data(sign)
                
                # Brief pause between signs
                if self.collection_active:
                    time.sleep(SIGN_DELAY)
            
            print("\nData collection completed!")
            
        finally:
            self.stop_camera()
            self.hand_detector.cleanup()
    
    def _show_sign_instruction(self, sign):
        """Show instruction for the current sign."""
        instruction_frame = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
        
        # Draw background
        cv2.rectangle(instruction_frame, (0, 0), (CAMERA_WIDTH, CAMERA_HEIGHT), (0, 0, 0), -1)
        
        # Draw sign letter
        cv2.putText(instruction_frame, sign, (CAMERA_WIDTH//2 - 50, CAMERA_HEIGHT//2 - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 3)
        
        # Draw instruction text
        cv2.putText(instruction_frame, f"Show the sign for '{sign}'", 
                   (CAMERA_WIDTH//2 - 200, CAMERA_HEIGHT//2 + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.putText(instruction_frame, "Press any key to continue...", 
                   (CAMERA_WIDTH//2 - 150, CAMERA_HEIGHT//2 + 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.imshow('Sign Instruction', instruction_frame)
        cv2.waitKey(0)
        cv2.destroyWindow('Sign Instruction')
    
    def save_data(self, filename: str = None):
        """Save collected data to CSV file."""
        if not self.data:
            print("No data to save")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sign_data_{timestamp}.csv"
        
        filepath = os.path.join(DATA_DIR, filename)
        
        # Convert to DataFrame and save
        df = pd.DataFrame(self.data)
        df.to_csv(filepath, index=False)
        
        print(f"Data saved to {filepath}")
        print(f"Total samples collected: {len(self.data)}")
        
        # Show summary by sign
        sign_counts = df['sign'].value_counts()
        print("\nSamples per sign:")
        for sign, count in sign_counts.items():
            print(f"  {sign}: {count}")
    
    def load_existing_data(self, filename: str = None):
        """Load existing data to continue collection."""
        if filename is None:
            # Look for existing CSV files
            csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
            if not csv_files:
                print("No existing data files found")
                return
            
            filename = csv_files[-1]  # Use most recent file
        
        filepath = os.path.join(DATA_DIR, filename)
        if os.path.exists(filepath):
            # Skip empty or zero-byte files safely
            if os.path.getsize(filepath) == 0:
                print("Existing CSV is empty; starting a new session.")
                return
            try:
                df = pd.read_csv(filepath)
                if df.empty:
                    print("Existing CSV has no rows; starting a new session.")
                    return
                # Basic schema sanity check
                expected_cols = {"sign", "landmarks"}
                if not expected_cols.issubset(set(df.columns)):
                    print("Existing CSV schema unexpected; starting a new session.")
                    return
                self.data = df.to_dict('records')
                print(f"Loaded {len(self.data)} existing samples from {filepath}")
            except pd.errors.EmptyDataError:
                print("Existing CSV is empty/invalid; starting a new session.")
            except Exception as e:
                print(f"Could not load existing data ({e}); starting a new session.")
        else:
            print(f"File not found: {filepath}")

def main():
    """Main function to run the data collection."""
    collector = SignDataCollector()
    
    print("=== Sign Language Data Collection Tool ===")
    print("This tool will help you collect hand gesture data for sign language recognition.")
    print("Make sure you have good lighting and a clear background.")
    print()
    
    # Check for existing data
    collector.load_existing_data()
    
    try:
        # Start collection
        collector.collect_all_signs()
        
        # Save data
        if collector.data:
            collector.save_data()
        
    except KeyboardInterrupt:
        print("\nData collection interrupted by user")
        if collector.data:
            collector.save_data()
    
    except Exception as e:
        print(f"Error during data collection: {e}")
        if collector.data:
            collector.save_data()

if __name__ == "__main__":
    main()
