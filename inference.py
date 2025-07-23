"""
Object Detection FastAPI Server
This FastAPI server loads a TensorFlow SavedModel and provides an endpoint for object detection inference.
"""

import os
import io
import base64
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO
from urllib.request import urlopen
import cv2
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn

# Environment variables / Configuration
HEIGHT, WIDTH = 640, 640
INPUT_IMAGE_SIZE = (HEIGHT, WIDTH)
MIN_SCORE_THRESH = 0.50
MODEL_PATH = './exported_object_detection_model'  # Path to the exported model in current directory

# Category index for label mapping
CATEGORY_INDEX = {
    1: {
        'id': 1,
        'name': 'object'
    }
}

# FastAPI app initialization
app = FastAPI(
    title="Object Detection API",
    version="1.0.0",
    description="A FastAPI server for object detection using TensorFlow SavedModel. Detects objects in images and returns bounding boxes with confidence scores.",
    docs_url="/docs",  # Swagger UI
    redoc_url="/redoc"  # ReDoc alternative documentation
)

# Global model variable
model_fn = None

# Request/Response models
class DetectionRequest(BaseModel):
    image_url: str = "https://example.com/image.jpg"
    min_score_threshold: Optional[float] = MIN_SCORE_THRESH
    
    class Config:
        schema_extra = {
            "example": {
                "image_url": "https://media.istockphoto.com/id/1412238893/photo/spaghetti-pasta-fusilli-lasagna-packagings-on-shelf-at-supermarket-3d-illustration.jpg?s=612x612&w=0&k=20&c=kpU6rRZn5_KnsIroyka05xc-RiTkiRx43JERRN2UFnM=",
                "min_score_threshold": 0.5
            }
        }

class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

class Detection(BaseModel):
    class_name: str
    confidence: float
    bbox: BoundingBox

class DetectionResponse(BaseModel):
    detection_count: int
    image_base64: str
    detections: list[Detection]
    
    class Config:
        schema_extra = {
            "example": {
                "detection_count": 2,
                "image_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
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
        }

def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array and return original size.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
        path: the file path to the image or URL

    Returns:
        uint8 numpy array with shape (img_height, img_width, 3) and original (width, height)
    """
    image = None
    if path.startswith('http'):
        response = urlopen(path)
        image_data = response.read()
        image_data = BytesIO(image_data)
        image = Image.open(image_data)
    else:
        image_data = tf.io.gfile.GFile(path, 'rb').read()
        image = Image.open(BytesIO(image_data))

    (im_width, im_height) = image.size
    # Convert the image to RGB format and then to a numpy array
    return np.array(image.convert('RGB')), (im_width, im_height)


def resize_and_crop_image(image, input_image_size, padded_size=None, aug_scale_min=1.0, aug_scale_max=1.0):
    """Resize and crop image to the specified size."""
    if padded_size is None:
        padded_size = input_image_size
    
    # Convert tensor to numpy if needed
    if isinstance(image, tf.Tensor):
        image_np = image.numpy()
    else:
        image_np = image
    
    height, width = input_image_size
    original_height, original_width = image_np.shape[:2]
    
    # Calculate scale to fit the image within the target size
    scale = min(height / original_height, width / original_width)
    
    # Calculate new dimensions
    new_height = int(original_height * scale)
    new_width = int(original_width * scale)
    
    # Resize the image
    resized_image = cv2.resize(image_np, (new_width, new_height))
    
    # Create padded image
    padded_image = np.zeros((height, width, 3), dtype=image_np.dtype)
    
    # Calculate padding offsets to center the image
    y_offset = (height - new_height) // 2
    x_offset = (width - new_width) // 2
    
    # Place resized image in the center of padded image
    padded_image[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_image
    
    return tf.constant(padded_image), scale


def visualize_boxes_and_labels_on_image_array(image, boxes, classes, scores, category_index,
                                            use_normalized_coordinates=False, max_boxes_to_draw=200,
                                            min_score_thresh=0.5, agnostic_mode=False,
                                            instance_masks=None, line_thickness=2):
    """Draw bounding boxes and labels on image array."""
    image_pil = Image.fromarray(image)
    
    # Convert to RGB if needed
    if image_pil.mode != 'RGB':
        image_pil = image_pil.convert('RGB')
    
    height, width = image.shape[:2]
    
    # Create a copy of the image to draw on
    image_with_boxes = np.array(image_pil)
    
    # Draw boxes for detections above threshold
    for i in range(min(len(boxes), max_boxes_to_draw)):
        if scores[i] >= min_score_thresh:
            box = boxes[i]
            
            if use_normalized_coordinates:
                y1, x1, y2, x2 = box
                y1 = int(y1 * height)
                x1 = int(x1 * width)
                y2 = int(y2 * height)
                x2 = int(x2 * width)
            else:
                y1, x1, y2, x2 = [int(coord) for coord in box]
            
            # Draw rectangle
            cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), line_thickness)
            
            # Get class name
            class_id = int(classes[i])
            class_name = category_index.get(class_id, {}).get('name', f'class_{class_id}')
            
            # Draw label
            label = f'{class_name}: {scores[i]:.2f}'
            
            # Calculate text size and draw background
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # Draw label background
            cv2.rectangle(image_with_boxes, (x1, y1 - text_height - 5), 
                         (x1 + text_width, y1), (0, 255, 0), -1)
            
            # Draw text
            cv2.putText(image_with_boxes, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Update the original image array
    image[:] = image_with_boxes


def build_inputs_for_object_detection(image, input_image_size):
    """Builds Object Detection model inputs for serving."""
    image, scale = resize_and_crop_image(
        image,
        input_image_size,
        padded_size=input_image_size,
        aug_scale_min=1.0,
        aug_scale_max=1.0)
    return image, scale


def load_model(model_path):
    """Load the TensorFlow SavedModel."""
    try:
        imported_model = tf.saved_model.load(model_path)
        model_fn = imported_model.signatures['serving_default']
        print(f"Successfully loaded model from '{model_path}'")
        return model_fn
    except Exception as e:
        print(f"Error loading model from '{model_path}': {e}")
        return None


def run_inference(model_fn, image_url, min_score_thresh=MIN_SCORE_THRESH):
    """Run inference on an image from URL and return API response data."""
    
    # Load the image from the URL and get original dimensions
    original_image_np, original_size = load_image_into_numpy_array(image_url)
    original_width, original_height = original_size
    print(f"Loaded image with original size: {original_width}x{original_height}")

    # Preprocess the image for the model
    image, scale = build_inputs_for_object_detection(tf.constant(original_image_np, dtype=tf.uint8), INPUT_IMAGE_SIZE)
    image = tf.expand_dims(image, axis=0)
    image = tf.cast(image, dtype=tf.uint8)

    # Run inference
    print("Running inference...")
    result = model_fn(image)

    # Count detections above threshold
    detections_above_threshold = np.sum(result['detection_scores'][0].numpy() > min_score_thresh)
    print(f"Number of detections found (score > {min_score_thresh}): {detections_above_threshold}")

    # Visualize the results on the processed image
    image_np_with_detections = image[0].numpy()
    visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        result['detection_boxes'][0].numpy(),
        result['detection_classes'][0].numpy().astype(int),
        result['detection_scores'][0].numpy(),
        category_index=CATEGORY_INDEX,
        use_normalized_coordinates=False,
        max_boxes_to_draw=200,
        min_score_thresh=min_score_thresh,
        agnostic_mode=False,
        instance_masks=None,
        line_thickness=2)

    # Calculate padding and crop the visualized image to original aspect ratio
    model_height, model_width = INPUT_IMAGE_SIZE
    
    # Calculate the actual dimensions after scaling
    scaled_height = int(original_height * scale)
    scaled_width = int(original_width * scale)
    
    # Calculate padding that was added during preprocessing
    y_offset = (model_height - scaled_height) // 2
    x_offset = (model_width - scaled_width) // 2
    
    # Crop to remove padding and get back to original aspect ratio
    cropped_image_np = image_np_with_detections[y_offset:y_offset + scaled_height, 
                                               x_offset:x_offset + scaled_width, :]

    # Convert image to base64 for API response
    pil_image = Image.fromarray(cropped_image_np.astype('uint8'))
    buffer = BytesIO()
    pil_image.save(buffer, format='JPEG')
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # Prepare detection details
    scores = result['detection_scores'][0].numpy()
    classes = result['detection_classes'][0].numpy().astype(int)
    boxes = result['detection_boxes'][0].numpy()
    
    detections = []
    for i, (score, class_id, box) in enumerate(zip(scores, classes, boxes)):
        if score > min_score_thresh:
            class_name = CATEGORY_INDEX.get(class_id, {}).get('name', 'unknown')
            y1, x1, y2, x2 = box
            detections.append({
                'class_name': class_name,
                'confidence': float(score),
                'bbox': {
                    'x1': float(x1),
                    'y1': float(y1), 
                    'x2': float(x2),
                    'y2': float(y2)
                }
            })

    return {
        'detection_count': int(detections_above_threshold),
        'image_base64': image_base64,
        'detections': detections
    }


# FastAPI startup event to load the model
@app.on_event("startup")
async def startup_event():
    global model_fn
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model not found at '{MODEL_PATH}'. Please ensure the exported_object_detection_model directory is in the current directory.")
    
    model_fn = load_model(MODEL_PATH)
    if model_fn is None:
        raise RuntimeError("Failed to load the model")
    
    print("\n" + "="*60)
    print("🚀 Object Detection API Server Started!")
    print("="*60)
    print("Model loaded successfully!")
    print(f"📊 Model input size: {HEIGHT}x{WIDTH}")
    print(f"🎯 Default confidence threshold: {MIN_SCORE_THRESH}")
    print("\n📋 Available Endpoints:")
    print("  🌐 API Root:        http://localhost:8000/")
    print("  🔍 Object Detection: http://localhost:8000/detect")
    print("  ❤️  Health Check:    http://localhost:8000/health")
    print("\n📚 Interactive Documentation:")
    print("  📖 Swagger UI:      http://localhost:8000/docs")
    print("  📄 ReDoc:           http://localhost:8000/redoc")
    print("\n💡 Click on the Swagger UI link above to test the API!")
    print("="*60 + "\n")

# Health check endpoint
@app.get("/health", 
         summary="Health Check",
         description="Check if the API is running and the model is loaded properly.")
async def health_check():
    return {"status": "healthy", "model_loaded": model_fn is not None}

# Object detection endpoint
@app.post("/detect", 
          response_model=DetectionResponse,
          summary="Detect Objects",
          description="Run object detection on an image from a URL. Returns the number of detections, bounding boxes, and a base64-encoded image with drawn bounding boxes.")
async def detect_objects(request: DetectionRequest):
    try:
        if model_fn is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        # Run inference
        result = run_inference(model_fn, request.image_url, request.min_score_threshold)
        
        return DetectionResponse(**result)
        
    except Exception as e:
        print(f"Error during inference: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

# Root endpoint with API information
@app.get("/",
         summary="API Information",
         description="Get basic information about the Object Detection API and available endpoints.")
async def root():
    return {
        "message": "Object Detection API",
        "version": "1.0.0",
        "description": "FastAPI server for object detection using TensorFlow SavedModel",
        "model_info": {
            "input_size": f"{HEIGHT}x{WIDTH}",
            "default_threshold": MIN_SCORE_THRESH,
            "supported_classes": list(CATEGORY_INDEX.values())
        },
        "endpoints": {
            "POST /detect": "Run object detection on an image from URL",
            "GET /health": "Check API health status",
            "GET /docs": "Swagger UI API documentation",
            "GET /redoc": "ReDoc API documentation"
        }
    }


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🔧 Starting Object Detection API Server...")
    print("="*60)
    print("⏳ Loading model and starting server...")
    print("📡 Server will be available at: http://localhost:8000")
    print("📖 Swagger UI will be at: http://localhost:8000/docs")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
