# SKU Detection API

A lightweight FastAPI server for Stock Keeping Unit (SKU) identification using ONNX model (RetinaNet).

## About

This API detects and identifies products/SKUs in retail images. The model provides general SKU detection with a single detection class, suitable for:

- Retail inventory management
- Product recognition in shelf images
- Automated stock monitoring
- Package/product identification

**Note:** The current model supports general SKU detection only, not specific product classification. All detected items are identified as generic "products" (class_id: 1).

## Features

- SKU detection via URL or file upload
- ONNX model support for fast inference
- RESTful API with JSON responses
- Base64 encoded result images with bounding boxes drawn around detected products
- Product identification with confidence scores

## Installation

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Ensure your `model.onnx` file is in the root directory

## Usage

Start the server:

```bash
python app.py
```

The API will be available at `http://localhost:8000`

**Documentation:** Visit `http://localhost:8000/docs` for interactive Swagger UI

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

The API returns detected SKUs with bounding box coordinates, confidence scores, and class IDs for product identification:

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
- `num_detections`: Total count of identified products
- `image_base64`: Original image with red bounding boxes drawn around detected SKUs
- `class_id`: Product/SKU identifier for inventory management
