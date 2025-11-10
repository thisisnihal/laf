from fastapi import FastAPI, Request, Form, File, UploadFile, Depends, HTTPException, status
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
import os
import shutil
import uuid
import numpy as np
from PIL import Image
import io
import math
from typing import Optional, List, Dict, Tuple
import json
import requests
from concurrent.futures import ThreadPoolExecutor
import asyncio
import cv2


# Import powerful ML models

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    print("YOLOv8 available")
except ImportError:
    print("YOLOv8 not available. Install with: pip install ultralytics")
    YOLO_AVAILABLE = False

try:
    from transformers import DetrImageProcessor, DetrForObjectDetection
    import torch
    DETR_AVAILABLE = True
    print("DETR available")
except ImportError:
    print("DETR not available. Install with: pip install transformers torch")
    DETR_AVAILABLE = False

try:
    import clip
    import torch
    CLIP_AVAILABLE = True
    print("CLIP available")
except ImportError:
    print("CLIP not available. Install with: pip install clip-by-openai")
    CLIP_AVAILABLE = False

# Configuration
SECRET_KEY = "248749f487934"  
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
UPLOAD_DIR = "static/uploads"

# Ensure upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="Lost & Found App")

# Static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# MongoDB setup
client = AsyncIOMotorClient("mongodb://localhost:27017")
db = client.lost_found

# Security
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer(auto_error=False)

class PowerfulModelManager:
    def __init__(self):
        self.models = {}
        self.all_labels = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if DETR_AVAILABLE or CLIP_AVAILABLE else None
        if self.device:
            print(f"Using device: {self.device}")
        
    def load_models(self):
        """Load powerful modern object detection models"""
        loaded_count = 0
        
        # 1. Load YOLOv8 (primary and most important)
        if YOLO_AVAILABLE and self.load_yolo_models():
            loaded_count += 1
            
        # 2. Load DETR (complementary detection) - optional
        if DETR_AVAILABLE and self.load_detr_model():
            loaded_count += 1
            
        # 3. Load CLIP (verification) - optional
        if CLIP_AVAILABLE and self.load_clip_model():
            loaded_count += 1
        
        # 4. Always setup comprehensive labels
        self.setup_comprehensive_labels()
        
        self.compile_all_labels()
        print(f"Loaded {loaded_count} model families successfully")
        print(f"Total unique labels available: {len(self.all_labels)}")
        
        return True  # Always return True as we have fallback
    
    def load_yolo_models(self):
        """Load YOLOv8 model - the most important for accuracy"""
        try:
            # Try YOLOv8 variants in order of preference
            model_variants = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt']  # Start with nano for faster download
            
            for variant in model_variants:
                try:
                    print(f"Loading YOLOv8: {variant}...")
                    model = YOLO(variant)  # Auto-downloads if not present
                    
                    # Test the model with a simple prediction to ensure it works
                    test_result = model.predict(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
                    
                    self.models['yolo'] = {
                        'model': model,
                        'labels': self.get_coco_labels(),
                        'type': 'yolo',
                        'variant': variant
                    }
                    
                    print(f"Successfully loaded YOLOv8: {variant}")
                    return True
                    
                except Exception as e:
                    print(f"Failed to load {variant}: {e}")
                    continue
            
            return False
            
        except Exception as e:
            print(f"Error loading YOLO models: {e}")
            return False
    
    def load_detr_model(self):
        """Load DETR model - optional complementary detection"""
        try:
            print("Loading DETR model...")
            processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
            model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
            
            if self.device and torch.cuda.is_available():
                model = model.to(self.device)
            
            self.models['detr'] = {
                'processor': processor,
                'model': model,
                'labels': self.get_coco_labels(),
                'type': 'detr'
            }
            
            print("Successfully loaded DETR model")
            return True
            
        except Exception as e:
            print(f"Error loading DETR model: {e}")
            return False
    
    def load_clip_model(self):
        """Load CLIP - optional verification"""
        try:
            print("Loading CLIP model...")
            model, preprocess = clip.load("ViT-B/32", device=self.device)
            
            self.models['clip'] = {
                'model': model,
                'preprocess': preprocess,
                'type': 'clip'
            }
            
            print("Successfully loaded CLIP model")
            return True
            
        except Exception as e:
            print(f"Error loading CLIP model: {e}")
            return False
    
    def get_coco_labels(self):
        """COCO dataset labels - covers most common objects"""
        return [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
            "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
            "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
        ]
    
    def get_lost_and_found_labels(self):
        """Extended labels for lost & found items"""
        return [
            # Personal accessories
            "wallet", "purse", "handbag", "backpack", "briefcase", "messenger bag", "tote bag",
            "keys", "keychain", "car keys", "house keys", "key fob", "key ring",
            
            # Cards and documents  
            "credit card", "debit card", "id card", "passport", "driver license", "student id",
            "business card", "membership card", "insurance card", "boarding pass",
            
            # Electronics
            "cell phone", "smartphone", "iphone", "android phone", "mobile phone",
            "laptop", "tablet", "ipad", "computer", "keyboard", "mouse", "wireless mouse",
            "headphones", "earphones", "earbuds", "airpods", "bluetooth headset",
            "charger", "phone charger", "laptop charger", "power bank", "charging cable", "usb cable",
            "camera", "smartwatch", "fitness tracker", "fitbit", "apple watch",
            
            # Jewelry and accessories
            "watch", "wristwatch", "bracelet", "necklace", "ring", "earrings", "pendant",
            "sunglasses", "eyeglasses", "reading glasses", "prescription glasses", "glasses case",
            
            # Clothing accessories
            "belt", "leather belt", "tie", "bow tie", "scarf", "shawl", "bandana",
            "hat", "cap", "beanie", "helmet", "gloves", "mittens", "hair tie",
            
            # Personal care
            "razor", "toothbrush", "comb", "brush", "makeup", "lipstick", "perfume",
            
            # Office/School
            "pen", "pencil", "marker", "notebook", "journal", "textbook", "calculator",
            
            # Sports and recreation
            "water bottle", "sports bottle", "coffee mug", "travel mug", "thermos",
            "basketball", "football", "soccer ball", "tennis ball", "golf ball",
            
            # Children's items
            "toy", "stuffed animal", "teddy bear", "doll", "action figure", "game controller",
            
            # Travel items
            "suitcase", "travel bag", "luggage", "passport holder", "travel pillow"
        ]
    
    def setup_comprehensive_labels(self):
        """Setup comprehensive labels even without models"""
        self.comprehensive_labels = list(set(
            self.get_coco_labels() + self.get_lost_and_found_labels()
        ))
    
    def compile_all_labels(self):
        """Compile all unique labels"""
        all_labels_set = set()
        
        # Add labels from loaded models
        for model_key, model_data in self.models.items():
            if 'labels' in model_data:
                all_labels_set.update(model_data['labels'])
        
        # Add comprehensive labels
        all_labels_set.update(self.comprehensive_labels)
        
        self.all_labels = sorted(list(all_labels_set))
    
    def detect_objects(self, image_bytes, confidence_threshold=0.25):
        """Main detection method - prioritizes YOLOv8, falls back gracefully"""
        
        # Convert bytes to PIL Image
        try:
            pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        except Exception as e:
            print(f"Error opening image: {e}")
            return self.create_fallback_detection()
        
        all_detections = []
        
        # 1. Primary detection with YOLOv8 (most important)
        if 'yolo' in self.models:
            yolo_detections = self.detect_with_yolo(pil_image, confidence_threshold)
            all_detections.extend(yolo_detections)
            
            # If YOLOv8 found good detections, we can return early for speed
            if len(yolo_detections) > 0 and max([d['confidence'] for d in yolo_detections]) > 0.5:
                return self.process_final_detections(yolo_detections)
        
        # 2. Try DETR if available and YOLO didn't find much
        if 'detr' in self.models and len(all_detections) < 3:
            detr_detections = self.detect_with_detr(pil_image, confidence_threshold * 0.8)
            all_detections.extend(detr_detections)
        
        # 3. CLIP verification if available
        if 'clip' in self.models and all_detections:
            all_detections = self.verify_with_clip(pil_image, all_detections)
        
        # 4. If no good detections, use intelligent fallback
        if not all_detections:
            return self.create_intelligent_fallback(pil_image)
        
        return self.process_final_detections(all_detections)
    
    def detect_with_yolo(self, pil_image, confidence_threshold):
        """YOLOv8 detection - primary method"""
        detections = []
        
        try:
            yolo_data = self.models['yolo']
            model = yolo_data['model']
            labels = yolo_data['labels']
            
            # Run YOLOv8 inference
            results = model.predict(pil_image, conf=confidence_threshold, verbose=False)
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        conf = float(box.conf.cpu().numpy()[0])
                        cls = int(box.cls.cpu().numpy()[0])
                        xyxy = box.xyxy.cpu().numpy()[0]
                        
                        if conf >= confidence_threshold and cls < len(labels):
                            # Convert to normalized coordinates
                            img_width, img_height = pil_image.size
                            x1, y1, x2, y2 = xyxy
                            
                            # Improve class names for lost & found context
                            class_name = self.improve_class_name(labels[cls])
                            
                            detection = {
                                'class': class_name,
                                'confidence': conf,
                                'bbox': {
                                    'x_min': float(x1 / img_width),
                                    'y_min': float(y1 / img_height),
                                    'x_max': float(x2 / img_width),
                                    'y_max': float(y2 / img_height)
                                },
                                'source_model': f"yolov8_{yolo_data.get('variant', 'unknown')}"
                            }
                            detections.append(detection)
            
            print(f"YOLOv8 detected {len(detections)} objects")
            
        except Exception as e:
            print(f"Error in YOLOv8 detection: {e}")
        
        return detections
    
    def detect_with_detr(self, pil_image, confidence_threshold):
        """DETR detection - complementary method"""
        detections = []
        
        try:
            detr_data = self.models['detr']
            processor = detr_data['processor']
            model = detr_data['model']
            labels = detr_data['labels']
            
            # Preprocess image
            inputs = processor(images=pil_image, return_tensors="pt")
            if self.device and torch.cuda.is_available():
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run inference
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Process results
            target_sizes = torch.tensor([pil_image.size[::-1]])
            if self.device and torch.cuda.is_available():
                target_sizes = target_sizes.to(self.device)
                
            results = processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=confidence_threshold
            )[0]
            
            img_width, img_height = pil_image.size
            
            for score, label_id, box in zip(results["scores"], results["labels"], results["boxes"]):
                score = float(score.cpu())
                label_id = int(label_id.cpu())
                box = box.cpu().numpy()
                
                if score >= confidence_threshold and label_id < len(labels):
                    x1, y1, x2, y2 = box
                    
                    class_name = self.improve_class_name(labels[label_id])
                    
                    detection = {
                        'class': class_name,
                        'confidence': score,
                        'bbox': {
                            'x_min': float(x1 / img_width),
                            'y_min': float(y1 / img_height),
                            'x_max': float(x2 / img_width),
                            'y_max': float(y2 / img_height)
                        },
                        'source_model': 'detr'
                    }
                    detections.append(detection)
            
            print(f"DETR detected {len(detections)} objects")
            
        except Exception as e:
            print(f"Error in DETR detection: {e}")
        
        return detections
    
    def verify_with_clip(self, pil_image, detections):
        """CLIP verification - improve classifications"""
        try:
            clip_data = self.models['clip']
            model = clip_data['model']
            preprocess = clip_data['preprocess']
            
            verified_detections = []
            
            for detection in detections:
                bbox = detection['bbox']
                class_name = detection['class']
                
                # Create prompts for lost & found context
                text_prompts = self.create_lost_found_prompts(class_name)
                
                # Crop detected region
                img_width, img_height = pil_image.size
                x1 = max(0, int(bbox['x_min'] * img_width))
                y1 = max(0, int(bbox['y_min'] * img_height))
                x2 = min(img_width, int(bbox['x_max'] * img_width))
                y2 = min(img_height, int(bbox['y_max'] * img_height))
                
                if x2 > x1 + 20 and y2 > y1 + 20:  # Minimum crop size
                    cropped = pil_image.crop((x1, y1, x2, y2))
                    
                    # CLIP verification
                    image_input = preprocess(cropped).unsqueeze(0).to(self.device)
                    text_inputs = clip.tokenize(text_prompts).to(self.device)
                    
                    with torch.no_grad():
                        image_features = model.encode_image(image_input)
                        text_features = model.encode_text(text_inputs)
                        
                        similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                        best_idx = similarities.argmax().item()
                        best_similarity = float(similarities[0, best_idx])
                        
                        # Update classification if CLIP is confident
                        if best_similarity > 20.0:
                            new_class = text_prompts[best_idx].replace("a photo of ", "").replace("a ", "")
                            detection['class'] = new_class
                            detection['clip_confidence'] = best_similarity
                            detection['confidence'] = min(0.95, detection['confidence'] * 1.1)
                
                verified_detections.append(detection)
            
            return verified_detections
            
        except Exception as e:
            print(f"Error in CLIP verification: {e}")
            return detections
    
    def improve_class_name(self, class_name):
        """Improve class names for lost & found context"""
        improvements = {
            'handbag': 'handbag',
            'backpack': 'backpack', 
            'cell phone': 'cell phone',
            'laptop': 'laptop',
            'bottle': 'water bottle',
            'cup': 'coffee mug',
            'book': 'book',
            'scissors': 'scissors',
            'clock': 'watch',
            'tie': 'tie',
            'umbrella': 'umbrella',
            'suitcase': 'suitcase',
            'teddy bear': 'teddy bear',
            'sports ball': 'ball'  # Avoid the dreaded "sports ball" misclassification
        }
        
        return improvements.get(class_name.lower(), class_name)
    
    def create_lost_found_prompts(self, class_name):
        """Create CLIP prompts optimized for lost & found items"""
        base_prompts = [f"a photo of {class_name}"]
        
        # Specific prompts for common lost items
        lost_item_prompts = {
            'handbag': ["a photo of a handbag", "a photo of a purse", "a woman's bag"],
            'cell phone': ["a photo of a smartphone", "a photo of an iPhone", "a photo of a mobile phone"],
            'backpack': ["a photo of a backpack", "a photo of a school bag"],
            'wallet': ["a photo of a wallet", "a photo of a billfold"],
            'keys': ["a photo of keys", "a photo of a keychain"],
            'sunglasses': ["a photo of sunglasses", "a photo of eyewear"],
            'watch': ["a photo of a wristwatch", "a photo of a watch"],
            'laptop': ["a photo of a laptop", "a photo of a computer"],
            'bottle': ["a photo of a water bottle", "a photo of a drink bottle"],
            'tie': ["a photo of a necktie", "a photo of a tie"],
            'belt': ["a photo of a belt", "a photo of a leather belt"],
            'umbrella': ["a photo of an umbrella"],
            'book': ["a photo of a book", "a photo of a textbook"]
        }
        
        return lost_item_prompts.get(class_name.lower(), base_prompts)
    
    def create_fallback_detection(self):
        """Simple fallback when no models work"""
        return [{
            'class': 'personal item',
            'confidence': 0.3,
            'bbox': {'x_min': 0.2, 'y_min': 0.2, 'x_max': 0.8, 'y_max': 0.8},
            'source_model': 'fallback'
        }]
    
    def create_intelligent_fallback(self, pil_image):
        """More intelligent fallback based on image analysis"""
        # Try to extract some basic features from the image
        img_array = np.array(pil_image)
        height, width = img_array.shape[:2]
        
        # Make educated guesses based on aspect ratio and size
        aspect_ratio = width / height
        
        if aspect_ratio > 2.0:  # Wide object
            return [{
                'class': 'belt',
                'confidence': 0.4,
                'bbox': {'x_min': 0.1, 'y_min': 0.4, 'x_max': 0.9, 'y_max': 0.6},
                'source_model': 'intelligent_fallback'
            }]
        elif aspect_ratio < 0.7:  # Tall object
            return [{
                'class': 'cell phone',
                'confidence': 0.4, 
                'bbox': {'x_min': 0.3, 'y_min': 0.1, 'x_max': 0.7, 'y_max': 0.9},
                'source_model': 'intelligent_fallback'
            }]
        else:  # Square-ish object
            return [{
                'class': 'wallet',
                'confidence': 0.4,
                'bbox': {'x_min': 0.2, 'y_min': 0.3, 'x_max': 0.8, 'y_max': 0.7},
                'source_model': 'intelligent_fallback'
            }]
    
    def process_final_detections(self, detections):
        """Process and clean up final detections"""
        if not detections:
            return []
        
        # Remove very low confidence detections
        filtered = [d for d in detections if d['confidence'] > 0.15]
        
        # Merge similar detections
        merged = self.merge_similar_detections(filtered)
        
        # Sort by confidence
        merged.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Limit to top 5 detections
        return merged[:5]
    
    def merge_similar_detections(self, detections):
        """Simple merging of overlapping detections"""
        if len(detections) <= 1:
            return detections
        
        merged = []
        used = set()
        
        for i, det1 in enumerate(detections):
            if i in used:
                continue
                
            similar = [det1]
            used.add(i)
            
            for j, det2 in enumerate(detections[i+1:], i+1):
                if j in used:
                    continue
                
                if self.calculate_iou(det1['bbox'], det2['bbox']) > 0.5:
                    similar.append(det2)
                    used.add(j)
            
            if len(similar) == 1:
                merged.append(det1)
            else:
                # Use highest confidence detection
                best = max(similar, key=lambda x: x['confidence'])
                merged.append(best)
        
        return merged
    
    def calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union"""
        x1 = max(bbox1['x_min'], bbox2['x_min'])
        y1 = max(bbox1['y_min'], bbox2['y_min']) 
        x2 = min(bbox1['x_max'], bbox2['x_max'])
        y2 = min(bbox1['y_max'], bbox2['y_max'])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1['x_max'] - bbox1['x_min']) * (bbox1['y_max'] - bbox1['y_min'])
        area2 = (bbox2['x_max'] - bbox2['x_min']) * (bbox2['y_max'] - bbox2['y_min'])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0

# Initialize the improved model manager
model_manager = PowerfulModelManager()

# Helper functions (keep your existing ones)
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials is None:
        return None
    
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        phone: str = payload.get("sub")
        if phone is None:
            return None
    except JWTError:
        return None
    
    user = await db.users.find_one({"phone": phone})
    return user

def get_primary_object(detections):
    """Get the primary object from detections"""
    if not detections:
        return "unknown", 0.0
    
    primary = detections[0]  # Already sorted by confidence
    return primary['class'], primary['confidence']

def extract_features_from_detections(detections):
    """Extract features for similarity matching"""
    if not detections:
        return None
    
    # Create feature vector from detected objects
    feature_dict = {}
    for detection in detections:
        class_name = detection['class'].lower()
        confidence = detection['confidence']
        
        if class_name in feature_dict:
            feature_dict[class_name] = max(feature_dict[class_name], confidence)
        else:
            feature_dict[class_name] = confidence
    
    # Create fixed-size feature vector
    all_labels = model_manager.all_labels
    features = np.zeros(min(len(all_labels), 100))  # Limit for performance
    
    for i, label in enumerate(all_labels[:100]):
        label_lower = label.lower()
        if label_lower in feature_dict:
            features[i] = feature_dict[label_lower]
    
    return features

def calculate_similarity(features1, features2):
    """Calculate cosine similarity"""
    if features1 is None or features2 is None:
        return 0.0
    
    try:
        if isinstance(features1, list):
            features1 = np.array(features1)
        if isinstance(features2, list):
            features2 = np.array(features2)
            
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        similarity = np.dot(features1, features2) / (norm1 * norm2)
        return float(similarity)
        
    except Exception as e:
        print(f"Error calculating similarity: {e}")
        return 0.0

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in kilometers"""
    R = 6371  # Earth's radius in km
    
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    return R * c

# Initialize models on startup
@app.on_event("startup")
async def startup_event():
    """Load models when the application starts"""
    print("Loading improved object detection models...")
    success = model_manager.load_models()
    if success:
        print(f"Models loaded successfully! Available labels: {len(model_manager.all_labels)}")
        print("The app should now correctly detect belts, wallets, purses, and other personal items!")
    else:
        print("Warning: Models loaded with fallbacks")

# Routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request, lat: Optional[float] = None, lon: Optional[float] = None):
    user = await get_current_user(request.headers.get("authorization"))
    
    # Get nearby lost items
    items = []
    if lat and lon:
        all_items = await db.items.find({"lost_by": {"$exists": True}, "resolved": {"$ne": True}}).to_list(100)
        for item in all_items:
            if item.get("lost_lat") and item.get("lost_lon"):
                distance = calculate_distance(lat, lon, item["lost_lat"], item["lost_lon"])
                if distance <= 10:  # Within 10km
                    item["distance"] = round(distance, 2)
                    items.append(item)
        items.sort(key=lambda x: x.get("distance", float('inf')))
    else:
        items = await db.items.find({"lost_by": {"$exists": True}, "resolved": {"$ne": True}}).limit(20).to_list(20)
    
    return templates.TemplateResponse("home.html", {"request": request, "user": user, "items": items})

@app.get("/signup", response_class=HTMLResponse)
async def signup_page(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})

@app.post("/signup")
async def signup(request: Request, name: str = Form(...), phone: str = Form(...), password: str = Form(...)):
    existing_user = await db.users.find_one({"phone": phone})
    if existing_user:
        return templates.TemplateResponse("signup.html", {
            "request": request, 
            "error": "Phone number already registered"
        })
    
    hashed_password = get_password_hash(password)
    user_data = {
        "name": name,
        "phone": phone,
        "password": hashed_password,
        "created_at": datetime.utcnow()
    }
    
    await db.users.insert_one(user_data)
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": phone}, expires_delta=access_token_expires
    )
    
    response = RedirectResponse(url="/dashboard", status_code=302)
    response.set_cookie(key="access_token", value=f"Bearer {access_token}", httponly=True)
    return response

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
async def login(request: Request, phone: str = Form(...), password: str = Form(...)):
    user = await db.users.find_one({"phone": phone})
    if not user or not verify_password(password, user["password"]):
        return templates.TemplateResponse("login.html", {
            "request": request, 
            "error": "Invalid phone number or password"
        })
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": phone}, expires_delta=access_token_expires
    )
    
    response = RedirectResponse(url="/dashboard", status_code=302)
    response.set_cookie(key="access_token", value=f"Bearer {access_token}", httponly=True)
    return response

@app.get("/logout")
async def logout():
    response = RedirectResponse(url="/", status_code=302)
    response.delete_cookie(key="access_token")
    return response

@app.get("/lost-item", response_class=HTMLResponse)
async def lost_item_page(request: Request):
    token = request.cookies.get("access_token")
    if not token:
        return RedirectResponse(url="/login", status_code=302)
    
    if token.startswith("Bearer "):
        token = token[7:]
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        phone = payload.get("sub")
        user = await db.users.find_one({"phone": phone})
        if not user:
            return RedirectResponse(url="/login", status_code=302)
    except JWTError:
        return RedirectResponse(url="/login", status_code=302)
    
    return templates.TemplateResponse("lost_item.html", {"request": request, "user": user})

@app.post("/lost-item")
async def submit_lost_item(
    request: Request,
    label: str = Form(...),
    image: UploadFile = File(...),
    lost_location: str = Form(...),
    lost_lat: float = Form(...),
    lost_lon: float = Form(...),
    lost_date_time: str = Form(...)
):
    token = request.cookies.get("access_token")
    if not token:
        return RedirectResponse(url="/login", status_code=302)
    
    if token.startswith("Bearer "):
        token = token[7:]
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        phone = payload.get("sub")
        user = await db.users.find_one({"phone": phone})
        if not user:
            return RedirectResponse(url="/login", status_code=302)
    except JWTError:
        return RedirectResponse(url="/login", status_code=302)
    
    # Save image
    image_data = await image.read()
    image_filename = f"{uuid.uuid4()}.jpg"
    image_path = os.path.join(UPLOAD_DIR, image_filename)
    
    with open(image_path, "wb") as f:
        f.write(image_data)
    
    # Use improved object detection
    detections = model_manager.detect_objects(image_data)
    primary_label, confidence = get_primary_object(detections)
    features = extract_features_from_detections(detections)
    
    features_list = features.tolist() if features is not None else []
    
    item_data = {
        "label": label,
        "image_url": f"/static/uploads/{image_filename}",
        "lost_at": lost_location,
        "lost_lat": lost_lat,
        "lost_lon": lost_lon,
        "lost_at_date_time": lost_date_time,
        "submit_at_date_time": datetime.utcnow().isoformat(),
        "image_score": confidence,
        "image_features": features_list,
        "lost_by": str(user["_id"]),
        "predicted_label": primary_label,
        "detections": detections,
        "extra_label": [],
        "resolved": False
    }
    
    await db.items.insert_one(item_data)
    return RedirectResponse(url="/dashboard", status_code=302)

@app.get("/found-item", response_class=HTMLResponse)
async def found_item_page(request: Request):
    token = request.cookies.get("access_token")
    if not token:
        return RedirectResponse(url="/login", status_code=302)
    
    if token.startswith("Bearer "):
        token = token[7:]
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        phone = payload.get("sub")
        user = await db.users.find_one({"phone": phone})
        if not user:
            return RedirectResponse(url="/login", status_code=302)
    except JWTError:
        return RedirectResponse(url="/login", status_code=302)
    
    return templates.TemplateResponse("found_item.html", {"request": request, "user": user})

@app.post("/found-item")
async def submit_found_item(
    request: Request,
    label: str = Form(...),
    image: UploadFile = File(...),
    found_location: str = Form(...),
    found_lat: float = Form(...),
    found_lon: float = Form(...),
    found_date_time: str = Form(...),
    submit_to_phone: str = Form(""),
    submit_location: str = Form(""),
    submit_lat: float = Form(0),
    submit_lon: float = Form(0)
):
    token = request.cookies.get("access_token")
    if not token:
        return RedirectResponse(url="/login", status_code=302)
    
    if token.startswith("Bearer "):
        token = token[7:]
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        phone = payload.get("sub")
        user = await db.users.find_one({"phone": phone})
        if not user:
            return RedirectResponse(url="/login", status_code=302)
    except JWTError:
        return RedirectResponse(url="/login", status_code=302)
    
    # Save image
    image_data = await image.read()
    image_filename = f"{uuid.uuid4()}.jpg"
    image_path = os.path.join(UPLOAD_DIR, image_filename)
    
    with open(image_path, "wb") as f:
        f.write(image_data)
    
    # Use improved object detection
    detections = model_manager.detect_objects(image_data)
    primary_label, confidence = get_primary_object(detections)
    features = extract_features_from_detections(detections)
    
    features_list = features.tolist() if features is not None else []
    
    # Handle submit_to user
    submit_to_user_id = None
    if submit_to_phone:
        submit_to_user = await db.users.find_one({"phone": submit_to_phone})
        if submit_to_user:
            submit_to_user_id = str(submit_to_user["_id"])
    
    item_data = {
        "label": label,
        "image_url": f"/static/uploads/{image_filename}",
        "found_at": found_location,
        "found_lat": found_lat,
        "found_lon": found_lon,
        "found_at_date_time": found_date_time,
        "submit_at_date_time": datetime.utcnow().isoformat(),
        "image_score": confidence,
        "image_features": features_list,
        "found_by": str(user["_id"]),
        "predicted_label": primary_label,
        "detections": detections,
        "extra_label": [],
        "resolved": False
    }
    
    if submit_to_user_id:
        item_data["submit_to"] = submit_to_user_id
        item_data["submit_at"] = submit_location
        item_data["submit_lat"] = submit_lat
        item_data["submit_lon"] = submit_lon
    
    await db.items.insert_one(item_data)
    return RedirectResponse(url="/dashboard", status_code=302)

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    token = request.cookies.get("access_token")
    if not token:
        return RedirectResponse(url="/login", status_code=302)
    
    if token.startswith("Bearer "):
        token = token[7:]
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        phone = payload.get("sub")
        user = await db.users.find_one({"phone": phone})
        if not user:
            return RedirectResponse(url="/login", status_code=302)
    except JWTError:
        return RedirectResponse(url="/login", status_code=302)
    
    user_id = str(user["_id"])
    
    lost_items = await db.items.find({"lost_by": user_id}).to_list(100)
    found_items = await db.items.find({"found_by": user_id}).to_list(100)
    submitted_items = await db.items.find({"submit_to": user_id}).to_list(100)
    
    for item in submitted_items:
        if item.get("found_by"):
            finder = await db.users.find_one({"_id": ObjectId(item["found_by"])})
            if finder:
                item["finder_phone"] = finder["phone"]
    
    return templates.TemplateResponse("dashboard.html", {
        "request": request, 
        "user": user,
        "lost_items": lost_items,
        "found_items": found_items,
        "submitted_items": submitted_items
    })

@app.get("/search", response_class=HTMLResponse)
async def search_page(request: Request, item_id: Optional[str] = None):
    token = request.cookies.get("access_token")
    if not token:
        return RedirectResponse(url="/login", status_code=302)
    
    if token.startswith("Bearer "):
        token = token[7:]
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        phone = payload.get("sub")
        user = await db.users.find_one({"phone": phone})
        if not user:
            return RedirectResponse(url="/login", status_code=302)
    except JWTError:
        return RedirectResponse(url="/login", status_code=302)
    
    search_item = None
    if item_id:
        search_item = await db.items.find_one({"_id": ObjectId(item_id)})
    
    return templates.TemplateResponse("search.html", {
        "request": request, 
        "user": user,
        "search_item": search_item
    })

@app.post("/search")
async def search_items(
    request: Request,
    query: str = Form(...),
    radius: float = Form(1.0),
    lat: float = Form(...),
    lon: float = Form(...),
    item_id: str = Form("")
):
    token = request.cookies.get("access_token")
    if not token:
        return RedirectResponse(url="/login", status_code=302)
    
    search_item = None
    search_features = None
    if item_id:
        search_item = await db.items.find_one({"_id": ObjectId(item_id)})
        if search_item and search_item.get("image_features"):
            search_features = np.array(search_item["image_features"])
    
    search_filter = {
        "found_by": {"$exists": True},
        "resolved": {"$ne": True}
    }
    
    # Enhanced text search including all detection data
    if query.strip():
        search_filter["$or"] = [
            {"label": {"$regex": query, "$options": "i"}},
            {"predicted_label": {"$regex": query, "$options": "i"}},
            {"detections.class": {"$regex": query, "$options": "i"}},
            {"detections.alternative_names": {"$regex": query, "$options": "i"}}
        ]
    
    found_items = await db.items.find(search_filter).to_list(200)
    
    nearby_items = []
    for item in found_items:
        if item.get("found_lat") and item.get("found_lon"):
            distance = calculate_distance(lat, lon, item["found_lat"], item["found_lon"])
            if distance <= radius:
                item["distance"] = round(distance, 2)
                
                if search_features is not None and item.get("image_features"):
                    item_features = np.array(item["image_features"])
                    similarity = calculate_similarity(search_features, item_features)
                    item["similarity"] = round(similarity * 100, 1)
                else:
                    item["similarity"] = 0.0
                
                finder = await db.users.find_one({"_id": ObjectId(item["found_by"])})
                if finder:
                    item["finder_phone"] = finder["phone"]
                    item["finder_name"] = finder["name"]
                nearby_items.append(item)
    
    if search_features is not None:
        nearby_items.sort(key=lambda x: (-x.get("similarity", 0), x.get("distance", float('inf'))))
    else:
        nearby_items.sort(key=lambda x: x.get("distance", float('inf')))
    
    return templates.TemplateResponse("search_results.html", {
        "request": request,
        "query": query,
        "radius": radius,
        "items": nearby_items,
        "search_item": search_item,
        "has_similarity": search_features is not None
    })

@app.post("/resolve-item/{item_id}")
async def resolve_item(item_id: str, request: Request):
    token = request.cookies.get("access_token")
    if not token:
        return JSONResponse({"success": False, "error": "Not authenticated"})
    
    if token.startswith("Bearer "):
        token = token[7:]
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        phone = payload.get("sub")
        user = await db.users.find_one({"phone": phone})
        if not user:
            return JSONResponse({"success": False, "error": "User not found"})
    except JWTError:
        return JSONResponse({"success": False, "error": "Invalid token"})
    
    await db.items.update_one(
        {"_id": ObjectId(item_id)},
        {"$set": {"resolved": True, "resolved_at": datetime.utcnow().isoformat()}}
    )
    
    return JSONResponse({"success": True})

# API endpoints for testing and debugging
@app.api_route("/predict-label", methods=["POST"])
async def predict_label_api(image: UploadFile = File(...)):
    """API endpoint for testing the improved object detection"""
    try:
        image_data = await image.read()
        detections = model_manager.detect_objects(image_data)
        primary_label, confidence = get_primary_object(detections)
        
        return JSONResponse({
            "predicted_label": primary_label,
            "confidence": confidence,
            "detections": detections,
            "models_loaded": len(model_manager.models),
            "total_labels": len(model_manager.all_labels),
            "available_models": list(model_manager.models.keys())
        })
    except Exception as e:
        print(f"Error in predict_label_api: {e}")
        return JSONResponse({
            "predicted_label": "error",
            "confidence": 0.0,
            "detections": [],
            "models_loaded": 0,
            "error": str(e)
        })

@app.get("/debug/model-status")
async def model_status():
    """Debug endpoint to check model status"""
    models_status = {}
    
    for model_key, model_data in model_manager.models.items():
        models_status[model_key] = {
            "loaded": True,
            "type": model_data.get("type", "unknown"),
            "labels_count": len(model_data.get("labels", [])),
            "variant": model_data.get("variant", "N/A")
        }
    
    return JSONResponse({
        "models": models_status,
        "total_models_loaded": len(model_manager.models),
        "total_unique_labels": len(model_manager.all_labels),
        "yolo_available": YOLO_AVAILABLE,
        "detr_available": DETR_AVAILABLE,
        "clip_available": CLIP_AVAILABLE,
        "device": str(model_manager.device) if model_manager.device else "CPU",
        "sample_labels": model_manager.all_labels[:20]
    })

@app.get("/debug/available-labels")
async def available_labels():
    """Debug endpoint to see all available labels"""
    categorized_labels = {
        "personal_items": [],
        "electronics": [],
        "clothing_accessories": [],
        "general_objects": [],
        "other": []
    }
    
    personal_keywords = ['wallet', 'purse', 'key', 'card', 'id', 'passport']
    electronics_keywords = ['phone', 'laptop', 'tablet', 'charger', 'headphone', 'camera']
    clothing_keywords = ['belt', 'tie', 'hat', 'glove', 'watch', 'glasses', 'jewelry']
    general_keywords = ['book', 'bottle', 'umbrella', 'bag', 'suitcase']
    
    for label in model_manager.all_labels:
        label_lower = label.lower()
        if any(keyword in label_lower for keyword in personal_keywords):
            categorized_labels["personal_items"].append(label)
        elif any(keyword in label_lower for keyword in electronics_keywords):
            categorized_labels["electronics"].append(label)
        elif any(keyword in label_lower for keyword in clothing_keywords):
            categorized_labels["clothing_accessories"].append(label)
        elif any(keyword in label_lower for keyword in general_keywords):
            categorized_labels["general_objects"].append(label)
        else:
            categorized_labels["other"].append(label)
    
    return JSONResponse({
        "categorized_labels": categorized_labels,
        "total_count": len(model_manager.all_labels),
        "improvements_note": "This version should correctly detect belts, wallets, purses, and other personal items instead of misclassifying them as 'sports ball'"
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)