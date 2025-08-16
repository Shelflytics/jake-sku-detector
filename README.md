# SKU Detection Engine

A lightweight FastAPI server for Stock Keeping Unit (SKU) detection using quantized ONNX model (RetinaNet). This core detection engine provides the underlying ML inference capabilities for SKU detection workflows.

## About

This detection engine detects products/SKUs in retail images using a trained RetinaNet model. The engine provides general SKU detection with a single detection class, suitable for:

- Retail inventory management
- Product recognition in shelf images
- Automated stock monitoring
- Package/product detection

**Note:** This engine supports general SKU detection only, not specific product classification. All detected items are detected as generic "products" (class_id: 1).

**Architecture:** This service acts as the core detection engine and can be integrated with higher-level APIs (like `jake-detector-api`) for enhanced business logic and workflow management.

## Features

- Core SKU detection engine with quantized ONNX model
- Fast inference for real-time detection with optimized model size
- RESTful API endpoints for integration
- Base64 encoded result images with bounding boxes drawn around detected products
- Confidence scoring for detection quality assessment
- Direct integration ready for higher-level services

## Installation

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Ensure your `model.onnx` file is in the root directory

## Usage

Start the detection engine:

```bash
python app.py
```

The detection engine will be available at `http://localhost:8000`

**Documentation:** Visit `http://localhost:8000/docs` for interactive Swagger UI

**Integration:** This engine can be integrated with higher-level APIs for enhanced workflow management.

## API Endpoints

### POST /detect/url

Detect SKUs from image URL

```json
{
  "image_url": "https://example.com/shelf-image.jpg",
  "confidence_threshold": 0.3
}
```

### POST /detect/upload

Detect SKUs from uploaded image file

- `file`: Image file (multipart/form-data)
- `confidence_threshold`: Confidence threshold (optional, default: 0.3)

### GET /health

Health check endpoint

## Response Format

The API returns detected SKUs with bounding box coordinates, confidence scores, and class IDs for product detection:

```json
{
  "detections": [
    {
      "x1": 100,
      "y1": 200,
      "x2": 300,
      "y2": 400,
      "confidence": 0.85,
      "class_id": 1
    }
  ],
  "num_detections": 1,
  "image_base64": "base64-encoded-image-with-bounding-boxes"
}
```

**Fields:**

- `detections`: Array of detected SKUs with pixel coordinates
- `num_detections`: Total count of detected products
- `image_base64`: Original image with red bounding boxes drawn around detected SKUs
- `class_id`: Product/SKU detector for inventory management
