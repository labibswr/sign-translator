import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
from typing import Tuple, List
from .config import *
from .hand_utils import normalize_landmarks, augment_landmarks

class DataPreprocessor:
    def __init__(self, data_path: str = None):
        """
        Initialize the data preprocessor.
        
        Args:
            data_path: Path to the CSV data file
        """
        self.data_path = data_path or os.path.join(DATA_DIR, 'sign_data.csv')
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
        
    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and preprocess the sign language data.
        
        Returns:
            Tuple of (features, labels)
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        # Load data
        df = pd.read_csv(self.data_path)
        
        # Extract features and labels
        features = []
        labels = []
        
        for _, row in df.iterrows():
            # Parse landmark coordinates
            landmark_str = row['landmarks']
            landmarks = self._parse_landmarks(landmark_str)
            
            if landmarks is not None:
                features.append(landmarks)
                labels.append(row['sign'])
        
        features = np.array(features)
        labels = np.array(labels)
        
        return features, labels
    
    def _parse_landmarks(self, landmark_str: str) -> np.ndarray:
        """
        Parse landmark string from CSV to numpy array.
        
        Args:
            landmark_str: String representation of landmarks
            
        Returns:
            Numpy array of landmarks
        """
        try:
            # Remove brackets and split by commas
            clean_str = landmark_str.strip('[]')
            coords = [float(x.strip()) for x in clean_str.split(',')]
            
            # Reshape to (21, 3)
            if len(coords) == 63:  # 21 landmarks * 3 coordinates
                return np.array(coords).reshape(21, 3)
            else:
                return None
        except:
            return None
    
    def preprocess_features(self, features: np.ndarray) -> np.ndarray:
        """
        Preprocess features by normalizing and augmenting.
        
        Args:
            features: Raw landmark features
            
        Returns:
            Preprocessed features
        """
        processed_features = []
        
        for feature in features:
            # Normalize landmarks
            normalized = normalize_landmarks(feature)
            
            # Add original sample
            processed_features.append(normalized)
            
            # Add augmented samples
            for _ in range(2):  # Create 2 augmented versions
                augmented = augment_landmarks(normalized)
                processed_features.append(augmented)
        
        return np.array(processed_features)
    
    def encode_labels(self, labels: np.ndarray) -> np.ndarray:
        """
        Encode string labels to numerical values.
        
        Args:
            labels: String labels
            
        Returns:
            Encoded numerical labels
        """
        if not self.is_fitted:
            self.label_encoder.fit(labels)
            self.is_fitted = True
        
        return self.label_encoder.transform(labels)
    
    def decode_labels(self, encoded_labels: np.ndarray) -> np.ndarray:
        """
        Decode numerical labels back to string labels.
        
        Args:
            encoded_labels: Encoded numerical labels
            
        Returns:
            Decoded string labels
        """
        if not self.is_fitted:
            raise ValueError("Label encoder not fitted. Call encode_labels first.")
        
        return self.label_encoder.inverse_transform(encoded_labels)
    
    def split_data(self, features: np.ndarray, labels: np.ndarray, 
                   test_size: float = 0.2, val_size: float = 0.2) -> Tuple:
        """
        Split data into train, validation, and test sets.
        
        Args:
            features: Feature array
            labels: Label array
            test_size: Proportion of test set
            val_size: Proportion of validation set from remaining data
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            features, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        # Second split: separate validation set from remaining data
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def prepare_data(self) -> Tuple:
        """
        Complete data preparation pipeline.
        
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test, label_encoder)
        """
        print("Loading data...")
        features, labels = self.load_data()
        print(f"  Loaded {len(features)} features and {len(labels)} labels")
        assert len(features) == len(labels), f"Mismatch: {len(features)} features vs {len(labels)} labels"
        
        print("Preprocessing features...")
        processed_features = self.preprocess_features(features)
        print(f"  Processed features shape: {processed_features.shape}")
        
        print("Encoding labels...")
        encoded_labels = self.encode_labels(labels)
        print(f"  Encoded labels shape: {encoded_labels.shape}")
        
        # Replicate labels to match augmented features (3 samples per original: 1 original + 2 augmented)
        # Each label needs to be repeated 3 times to match the feature augmentation
        num_augmentations = 3  # 1 original + 2 augmented
        replicated_labels = np.repeat(encoded_labels, num_augmentations)
        print(f"  Replicated labels shape: {replicated_labels.shape}")
        print(f"  Shapes match: {processed_features.shape[0] == replicated_labels.shape[0]}")
        
        print("Splitting data...")
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(
            processed_features, replicated_labels
        )
        
        print(f"Data prepared:")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Validation samples: {len(X_val)}")
        print(f"  Test samples: {len(X_test)}")
        print(f"  Feature shape: {X_train.shape[1:]}")
        print(f"  Number of classes: {len(self.label_encoder.classes_)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test, self.label_encoder
    
    def save_preprocessed_data(self, output_path: str = None):
        """
        Save preprocessed data to files for faster loading.
        
        Args:
            output_path: Directory to save preprocessed data
        """
        if output_path is None:
            output_path = DATA_DIR
        
        os.makedirs(output_path, exist_ok=True)
        
        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test, label_encoder = self.prepare_data()
        
        # Save numpy arrays
        np.save(os.path.join(output_path, 'X_train.npy'), X_train)
        np.save(os.path.join(output_path, 'X_val.npy'), X_val)
        np.save(os.path.join(output_path, 'X_test.npy'), X_test)
        np.save(os.path.join(output_path, 'y_train.npy'), y_train)
        np.save(os.path.join(output_path, 'y_val.npy'), y_val)
        np.save(os.path.join(output_path, 'y_test.npy'), y_test)
        
        # Save label encoder classes
        np.save(os.path.join(output_path, 'label_classes.npy'), label_encoder.classes_)
        
        print(f"Preprocessed data saved to {output_path}")
    
    def load_preprocessed_data(self, data_path: str = None) -> Tuple:
        """
        Load preprocessed data from files.
        
        Args:
            data_path: Directory containing preprocessed data
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test, label_encoder)
        """
        if data_path is None:
            data_path = DATA_DIR
        
        # Load numpy arrays
        X_train = np.load(os.path.join(data_path, 'X_train.npy'))
        X_val = np.load(os.path.join(data_path, 'X_val.npy'))
        X_test = np.load(os.path.join(data_path, 'X_test.npy'))
        y_train = np.load(os.path.join(data_path, 'y_train.npy'))
        y_val = np.load(os.path.join(data_path, 'y_val.npy'))
        y_test = np.load(os.path.join(data_path, 'y_test.npy'))
        
        # Load label encoder classes
        classes = np.load(os.path.join(data_path, 'label_classes.npy'))
        self.label_encoder.classes_ = classes
        self.is_fitted = True
        
        print(f"Preprocessed data loaded from {data_path}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test, self.label_encoder
