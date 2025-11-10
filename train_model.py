import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime
from utils.config import *
from utils.preprocessing import DataPreprocessor

class SignLanguageModel:
    def __init__(self):
        """Initialize the sign language recognition model."""
        self.model = None
        self.history = None
        self.label_encoder = None
        
        # Create models directory if it doesn't exist
        os.makedirs(MODELS_DIR, exist_ok=True)
        
        # Set random seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)
    
    def build_model(self, input_shape: tuple = None, num_classes: int = None):
        """
        Build the neural network model.
        
        Args:
            input_shape: Shape of input features
            num_classes: Number of output classes
        """
        if input_shape is None:
            input_shape = INPUT_SHAPE
        if num_classes is None:
            num_classes = NUM_CLASSES
        
        # Flatten input shape for the model
        if len(input_shape) == 2:
            input_shape = (input_shape[0] * input_shape[1],)
        
        print(f"Building model with input shape: {input_shape}, output classes: {num_classes}")
        
        # Build the model
        self.model = keras.Sequential([
            # Input layer
            layers.Input(shape=input_shape),
            
            # Dense layers with dropout for regularization
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.3),
            layers.BatchNormalization(),
            
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.BatchNormalization(),
            
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.BatchNormalization(),
            
            # Output layer
            layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compile the model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Print model summary
        self.model.summary()
    
    def train_model(self, X_train, y_train, X_val, y_val, 
                   epochs: int = None, batch_size: int = None):
        """
        Train the model on the provided data.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        if epochs is None:
            epochs = EPOCHS
        if batch_size is None:
            batch_size = BATCH_SIZE
        
        print(f"Starting training with {epochs} epochs, batch size {batch_size}")
        
        # Callbacks for better training
        callbacks = [
            # Early stopping to prevent overfitting
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            # Reduce learning rate when plateau is reached
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            ),
            # Model checkpoint to save best model
            keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(MODELS_DIR, 'best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False
            )
        ]
        
        # Train the model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print("Training completed!")
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the trained model on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
        """
        if self.model is None:
            raise ValueError("Model not trained. Train the model first.")
        
        print("Evaluating model on test data...")
        
        # Get predictions
        y_pred_proba = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Calculate accuracy
        test_accuracy = np.mean(y_pred == y_test)
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                  target_names=self.label_encoder.classes_))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        self._plot_confusion_matrix(cm)
        
        return test_accuracy, y_pred, y_pred_proba
    
    def _plot_confusion_matrix(self, cm):
        """Plot confusion matrix."""
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(MODELS_DIR, 'confusion_matrix.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {plot_path}")
        plt.show()
    
    def plot_training_history(self):
        """Plot training history (loss and accuracy)."""
        if self.history is None:
            print("No training history available.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        ax1.plot(self.history.history['loss'], label='Training Loss')
        ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax2.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(MODELS_DIR, 'training_history.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Training history saved to {plot_path}")
        plt.show()
    
    def save_model(self, filename: str = None):
        """Save the trained model."""
        if self.model is None:
            print("No model to save.")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sign_model_{timestamp}.h5"
        
        filepath = os.path.join(MODELS_DIR, filename)
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
        
        # Also save label encoder info
        encoder_path = os.path.join(MODELS_DIR, 'label_encoder.npy')
        np.save(encoder_path, self.label_encoder.classes_)
        print(f"Label encoder saved to {encoder_path}")
    
    def load_model(self, model_path: str, encoder_path: str = None):
        """Load a trained model."""
        if encoder_path is None:
            encoder_path = os.path.join(MODELS_DIR, 'label_encoder.npy')
        
        # Load model
        self.model = keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")
        
        # Load label encoder
        classes = np.load(encoder_path)
        from sklearn.preprocessing import LabelEncoder
        self.label_encoder = LabelEncoder()
        self.label_encoder.classes_ = classes
        print(f"Label encoder loaded from {encoder_path}")
    
    def predict_sign(self, landmarks: np.ndarray) -> tuple:
        """
        Predict sign from hand landmarks.
        
        Args:
            landmarks: Hand landmark coordinates
            
        Returns:
            Tuple of (predicted_sign, confidence)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Load a trained model first.")
        
        # Preprocess landmarks
        if landmarks.shape != (1, -1):
            landmarks = landmarks.reshape(1, -1)
        
        # Make prediction
        prediction = self.model.predict(landmarks, verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class]
        
        # Decode label
        predicted_sign = self.label_encoder.inverse_transform([predicted_class])[0]
        
        return predicted_sign, confidence

def main():
    """Main function to train the sign language model."""
    print("=== Sign Language Model Training ===")
    
    # Initialize preprocessor and model
    preprocessor = DataPreprocessor()
    model = SignLanguageModel()
    
    try:
        # Check if preprocessed data exists
        preprocessed_files = ['X_train.npy', 'X_val.npy', 'X_test.npy', 
                            'y_train.npy', 'y_val.npy', 'y_test.npy']
        
        data_path = DATA_DIR
        if all(os.path.exists(os.path.join(data_path, f)) for f in preprocessed_files):
            print("Loading preprocessed data...")
            X_train, X_val, X_test, y_train, y_val, y_test, label_encoder = \
                preprocessor.load_preprocessed_data(data_path)
        else:
            print("Preprocessed data not found. Preparing data from CSV...")
            X_train, X_val, X_test, y_train, y_val, y_test, label_encoder = \
                preprocessor.prepare_data()
            
            # Save preprocessed data for future use
            preprocessor.save_preprocessed_data(data_path)
        
        # Store label encoder in model
        model.label_encoder = label_encoder
        
        # Build and train model
        print("\nBuilding model...")
        model.build_model()
        
        print("\nTraining model...")
        model.train_model(X_train, y_train, X_val, y_val)
        
        # Evaluate model
        print("\nEvaluating model...")
        test_accuracy, y_pred, y_pred_proba = model.evaluate_model(X_test, y_test)
        
        # Plot training history
        print("\nPlotting training history...")
        model.plot_training_history()
        
        # Save model
        print("\nSaving model...")
        model.save_model()
        
        print(f"\nTraining completed successfully!")
        print(f"Final test accuracy: {test_accuracy:.4f}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
