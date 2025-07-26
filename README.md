# Lightweight Object Detection API

A lightweight FastAPI server for object detection inference using TensorFlow SavedModel (RetinaNet).

## Features

- Object detection via URL or file upload
- TensorFlow SavedModel support
- RESTful API with JSON responses
- Base64 encoded result images with bounding boxes

## Installation

1. Install dependencies:

```bash
pip install -r requirements.txt
```

1. Ensure your model is in the `exported_object_detection_model/` directory

## Usage

Start the server:

```bash
python inference.py
```

The API will be available at `http://localhost:8000`

**Documentation:** Visit `http://localhost:8000/docs` for interactive Swagger UI

## API Endpoints

### POST /detect

Detect objects from image URL

```json
{
  "image_url": "https://example.com/image.jpg",
  "min_score_threshold": 0.5
}
```

### POST /detect-upload

Detect objects from uploaded image file

- `file`: Image file (multipart/form-data)
- `min_score_threshold`: Confidence threshold (optional)

### GET /health

Health check endpoint

## Response Format

```json
{
  "detection_count": 2,
  "image_base64": "base64-encoded-image-with-boxes",
  "detections": [
    {
      "class_name": "object",
      "confidence": 0.85,
      "bbox": {
        "x1": 100.5,
        "y1": 200.3,
        "x2": 300.7,
        "y2": 400.9
      }
    }
  ]
}
```
