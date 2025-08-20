import onnxruntime as ort
import numpy as np
import requests
from PIL import Image, ImageDraw
from io import BytesIO
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
import base64

# Configuration
MODEL_PATH = "model.onnx"
INPUT_SIZE = (640, 640)
CONFIDENCE_THRESHOLD = 0.3

# Initialize FastAPI
app = FastAPI(title="Object Detection API", version="1.0.0")

# Global model session
session = None

# Request/Response models
class DetectionRequest(BaseModel):
    image_url: str
    confidence_threshold: Optional[float] = CONFIDENCE_THRESHOLD

class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int

class DetectionResponse(BaseModel):
    detections: List[BoundingBox]
    num_detections: int
    image_base64: str

def load_model():
    """Load ONNX model"""
    global session
    if session is None:
        session = ort.InferenceSession(MODEL_PATH)
    return session

def preprocess_image(image: Image.Image):
    """Preprocess image for model input"""
    original_size = image.size
    image = image.convert("RGB")
    image = image.resize(INPUT_SIZE)
    
    # Convert to numpy array as uint8
    input_array = np.array(image, dtype=np.uint8)
    input_array = np.expand_dims(input_array, axis=0)  # Add batch dimension
    
    return input_array, original_size

def postprocess_results(results, original_size, confidence_threshold):
    """Convert model outputs to bounding boxes"""
    detection_boxes = results[0][0]  # [N, 4]
    detection_classes = results[1][0]  # [N]
    detection_scores = results[2][0]  # [N]
    
    detections = []
    scale_x = original_size[0] / INPUT_SIZE[0]
    scale_y = original_size[1] / INPUT_SIZE[1]
    
    for box, cls, score in zip(detection_boxes, detection_classes, detection_scores):
        if score >= confidence_threshold:
            ymin, xmin, ymax, xmax = box
            # Convert normalized coords to pixel coords using scale factors
            x1 = int(xmin * scale_x)
            y1 = int(ymin * scale_y)
            x2 = int(xmax * scale_x)
            y2 = int(ymax * scale_y)
            
            detections.append(BoundingBox(
                x1=x1, y1=y1, x2=x2, y2=y2,
                confidence=float(score),
                class_id=int(cls)
            ))
    
    return detections

def draw_bounding_boxes(image: Image.Image, detections: List[BoundingBox]) -> str:
    """Draw bounding boxes on image and return base64 encoded result"""
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)
    
    for detection in detections:
        # Draw rectangle
        draw.rectangle(
            [detection.x1, detection.y1, detection.x2, detection.y2],
            outline="red",
            width=2
        )
        
        # Draw confidence score
        text = f"{detection.confidence:.2f}"
        draw.text((detection.x1, detection.y1 - 15), text, fill="red")
    
    # Convert to base64
    buffer = BytesIO()
    draw_image.save(buffer, format="JPEG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    
    return img_base64

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()

@app.post("/detect/url", response_model=DetectionResponse)
async def detect_from_url(request: DetectionRequest):
    """Run inference on image from URL"""
    try:
        # Download image
        response = requests.get(request.image_url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        
        # Preprocess
        input_array, original_size = preprocess_image(image)
        
        # Run inference
        sess = load_model()
        results = sess.run(None, {"inputs": input_array})
        
        # Postprocess
        detections = postprocess_results(results, original_size, request.confidence_threshold)
        
        # Draw bounding boxes on original image
        image_base64 = draw_bounding_boxes(image, detections)
        
        return DetectionResponse(
            detections=detections,
            num_detections=len(detections),
            image_base64=image_base64
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/detect/upload", response_model=DetectionResponse)
async def detect_from_upload(
    file: UploadFile = File(...),
    confidence_threshold: Optional[float] = CONFIDENCE_THRESHOLD
):
    """Run inference on uploaded image file"""
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process image
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        
        # Preprocess
        input_array, original_size = preprocess_image(image)
        
        # Run inference
        sess = load_model()
        results = sess.run(None, {"inputs": input_array})
        
        # Postprocess
        detections = postprocess_results(results, original_size, confidence_threshold)
        
        # Draw bounding boxes on original image
        image_base64 = draw_bounding_boxes(image, detections)
        
        return DetectionResponse(
            detections=detections,
            num_detections=len(detections),
            image_base64=image_base64
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": session is not None}

if __name__ == "__main__":
    print("🚀 Starting SKU Detection API...")
    print("📋 API Documentation: http://localhost:5000/docs")
    print("🔄 Alternative Docs: http://localhost:5000/redoc") 
    print("❤️  Health Check: http://localhost:5000/health")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=5000)
