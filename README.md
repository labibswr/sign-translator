# Sign Language Translator

A real-time sign language recognition system that can translate American Sign Language (ASL) alphabet gestures into text using computer vision and machine learning.

## Features

- **Real-time Hand Detection**: Uses MediaPipe for accurate hand landmark detection
- **ASL Alphabet Support**: Recognizes all 26 letters (A-Z) of the American Sign Language alphabet
- **Live Translation**: Real-time sign language recognition from webcam feed
- **Data Collection Tool**: Interactive tool for collecting training data
- **Neural Network Model**: Deep learning model for accurate sign classification
- **Data Augmentation**: Automatic data augmentation to improve model robustness
- **Visual Feedback**: Real-time visual indicators and confidence scores

## Project Structure

```
sign-translator/
├── data/                   # Data storage directory
├── models/                 # Trained models and plots
├── utils/                  # Utility modules
│   ├── config.py          # Configuration settings
│   ├── hand_utils.py      # Hand detection utilities
│   └── preprocessing.py    # Data preprocessing utilities
├── data_collection.py      # Data collection script
├── train_model.py          # Model training script
├── predict_live.py         # Live prediction script
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Requirements

- Python 3.8+
- Webcam
- Good lighting conditions
- Clear background for hand detection

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd sign-translator
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```bash
   python -c "import cv2, mediapipe, tensorflow; print('All dependencies installed successfully!')"
   ```

## Usage

### 1. Data Collection

First, you need to collect training data for the signs you want to recognize:

```bash
python data_collection.py
```

**Instructions for data collection**:
- The tool will guide you through each letter of the alphabet
- Show the correct ASL sign for each letter
- Keep your hand centered in the camera frame
- Each sign will collect 100 samples by default
- Press 'q' to quit or 's' to skip a sign

**Tips for better data collection**:
- Ensure good lighting
- Use a plain background
- Keep your hand clearly visible
- Vary hand positions slightly for robustness
- Take breaks between signs if needed

### 2. Model Training

After collecting data, train the neural network model:

```bash
python train_model.py
```

**Training process**:
- Automatically loads and preprocesses collected data
- Splits data into training, validation, and test sets
- Applies data augmentation for better generalization
- Trains a deep neural network with early stopping
- Generates training plots and confusion matrix
- Saves the best model automatically

**Training parameters** (configurable in `utils/config.py`):
- Epochs: 100 (with early stopping)
- Batch size: 32
- Learning rate: 0.001
- Input shape: 21 landmarks × 3 coordinates

### 3. Live Prediction

Once training is complete, run live sign language recognition:

```bash
python predict_live.py
```

**Live recognition features**:
- Real-time hand detection and tracking
- Instant sign classification
- Confidence scores and visual feedback
- Prediction smoothing for stability
- Press 'q' to quit

## Configuration

You can customize various settings in `utils/config.py`:

- **Camera settings**: Resolution, FPS
- **Model parameters**: Learning rate, epochs, batch size
- **Data collection**: Samples per sign, delays
- **Detection settings**: Confidence thresholds
- **Supported signs**: Modify the signs list

## Model Architecture

The neural network consists of:
- **Input Layer**: 63 features (21 landmarks × 3 coordinates)
- **Hidden Layers**: 512 → 256 → 128 neurons with dropout and batch normalization
- **Output Layer**: 26 classes (A-Z) with softmax activation
- **Regularization**: Dropout (0.2-0.3) and batch normalization
- **Optimizer**: Adam with learning rate scheduling

## Data Format

The system uses MediaPipe's 21-point hand landmark model:
- **Landmarks**: 21 points on the hand (wrist, finger joints, tips)
- **Coordinates**: X, Y, Z coordinates for each landmark
- **Normalization**: Automatic translation and scale invariance
- **Augmentation**: Noise addition and rotation for robustness

## Troubleshooting

### Common Issues

1. **Camera not working**:
   - Check if webcam is connected and accessible
   - Try different camera indices (0, 1, 2...)
   - Ensure no other applications are using the camera

2. **Poor hand detection**:
   - Improve lighting conditions
   - Use a plain, contrasting background
   - Keep hand clearly visible and centered
   - Check camera focus and resolution

3. **Low prediction accuracy**:
   - Collect more training data
   - Ensure data quality and consistency
   - Retrain the model with more epochs
   - Check for overfitting in training plots

4. **Dependencies issues**:
   - Use Python 3.8+ for compatibility
   - Install CUDA if using GPU acceleration
   - Update pip: `pip install --upgrade pip`

### Performance Tips

- **GPU acceleration**: Install TensorFlow-GPU for faster training
- **Data quality**: Collect diverse, well-lit samples
- **Regular retraining**: Retrain periodically with new data
- **Model optimization**: Experiment with different architectures

## Extending the Project

### Adding New Signs

1. Modify `SIGNS` list in `utils/config.py`
2. Update `NUM_CLASSES` in configuration
3. Collect data for new signs
4. Retrain the model

### Custom Hand Gestures

1. Extend `HandDetector` class in `utils/hand_utils.py`
2. Add gesture-specific preprocessing
3. Modify data collection for custom gestures
4. Update model architecture if needed

### Integration

The system can be integrated into:
- Web applications
- Mobile apps
- Educational software
- Accessibility tools
- Communication platforms

## Contributing

Contributions are welcome! Areas for improvement:
- Additional sign language support
- Better data augmentation techniques
- Model architecture improvements
- User interface enhancements
- Performance optimizations

## License

This project is open source. Please check the license file for details.

## Acknowledgments

- **MediaPipe**: Hand landmark detection
- **TensorFlow**: Deep learning framework
- **OpenCV**: Computer vision library
- **ASL Community**: Sign language guidance

## Support

For questions or issues:
1. Check the troubleshooting section
2. Review the configuration options
3. Ensure all dependencies are installed
4. Check the console output for error messages

---

**Note**: This system is designed for educational and research purposes. For production use, consider additional validation, security measures, and accessibility compliance.
