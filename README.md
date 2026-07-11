# Face Mask Detection

A Flask-based web application for real-time face mask detection using Vision Transformer (ViT) model..

## Features

- Real-time face mask detection through web cam
- Image upload and processing
- Face detection and classification
- Web-based interface with live video streaming
- Automatic saving of faces without masks

## Model Performance

- **Accuracy**: 99.53%
- **Loss**: 0.0239

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python app.py
```

3. Open your browser and navigate to `http://localhost:5000`

## Current Framework Versions

- Flask: 3.0.0
- Transformers: 5.0.0
- PyTorch: 2.9.0
- OpenCV: 4.12.0.88
- NumPy: 2.2.6
- Pillow: 11.3.0
- Safetensors: 0.6.2

## Project Structure

```
Face mask detection/
├── app.py                 # Main Flask application
├── model.safetensors      # Trained model weights
├── haarcascade_frontalface_default.xml  # OpenCV face detection
├── preprocessor_config.json # Model preprocessing config
├── requirements.txt       # Python dependencies
├── static/               # CSS and JavaScript files
├── templates/            # HTML templates
└── uploads/              # User upload directory
```

## Usage

1. **Live Detection**: Click on "Live Detection" to start real-time mask detection through your webcam
2. **Image Upload**: Upload images for batch processing
3. **Results**: View detection results and download faces without masks (if any are detected)

## Model Details

- **Base Model**: google/vit-base-patch16-224-in21k
- **Task**: Image Classification (Mask vs No Mask)
- **Input Size**: 224x224 pixels
