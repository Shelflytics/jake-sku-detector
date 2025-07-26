"""
Lightweight Object Detection FastAPI Server
FastAPI server for object detection inference using TensorFlow SavedModel.
"""

import os
import base64
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO
from urllib.request import urlopen
import cv2
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
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
    description="Lightweight FastAPI server for object detection using TensorFlow SavedModel.",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global model variable
model_fn = None

# Request/Response models
class DetectionRequest(BaseModel):
    image_url: str
    min_score_threshold: Optional[float] = MIN_SCORE_THRESH

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

def load_image_into_numpy_array(path_or_file):
    """Load an image from file, URL, or uploaded file into a numpy array and return original size.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
        path_or_file: the file path to the image, URL, or UploadFile object

    Returns:
        uint8 numpy array with shape (img_height, img_width, 3) and original (width, height)
    """
    image = None
    
    # Handle UploadFile object
    if hasattr(path_or_file, 'file'):
        image_data = path_or_file.file.read()
        image = Image.open(BytesIO(image_data))
    # Handle URL
    elif isinstance(path_or_file, str) and path_or_file.startswith('http'):
        response = urlopen(path_or_file)
        image_data = response.read()
        image_data = BytesIO(image_data)
        image = Image.open(image_data)
    # Handle file path
    else:
        image_data = tf.io.gfile.GFile(path_or_file, 'rb').read()
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


def run_inference(model_fn, image_source, min_score_thresh=MIN_SCORE_THRESH):
    """Run inference on an image from URL, file path, or UploadFile and return API response data."""
    
    # Load the image from the source and get original dimensions
    original_image_np, original_size = load_image_into_numpy_array(image_source)
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
    
    print("Object Detection API Server Started!")
    print(f"Model input size: {HEIGHT}x{WIDTH}")
    print(f"Default confidence threshold: {MIN_SCORE_THRESH}")
    print("Available endpoints:")
    print("  POST /detect - Object detection from URL")
    print("  POST /detect-upload - Object detection from uploaded file")
    print("  GET /health - Health check")
    print("  GET /docs - Swagger UI documentation")
    print("  GET /redoc - ReDoc documentation")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model_fn is not None}

# Object detection endpoint for URL
@app.post("/detect", response_model=DetectionResponse)
async def detect_objects(request: DetectionRequest):
    try:
        if model_fn is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        result = run_inference(model_fn, request.image_url, request.min_score_threshold)
        return DetectionResponse(**result)
        
    except Exception as e:
        print(f"Error during inference: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

# Object detection endpoint for file upload
@app.post("/detect-upload", response_model=DetectionResponse)
async def detect_objects_upload(
    file: UploadFile = File(...),
    min_score_threshold: float = Form(MIN_SCORE_THRESH)
):
    try:
        if model_fn is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        result = run_inference(model_fn, file, min_score_threshold)
        return DetectionResponse(**result)
        
    except Exception as e:
        print(f"Error during inference: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


if __name__ == "__main__":
    print("Starting Object Detection API Server...")
    print("Server will be available at: http://localhost:8000")
    print("Swagger UI will be available at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
