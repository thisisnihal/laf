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
import uuid
import numpy as np
from PIL import Image
import io
import math
from typing import Optional, List, Dict
import json

# Import MobileNet for image classification
try:
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    MOBILENET_AVAILABLE = True
    print("MobileNet available")
except ImportError:
    print("MobileNet not available. Install with: pip install transformers torch pillow")
    MOBILENET_AVAILABLE = False

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


class MobileNetModelManager:
    def __init__(self):
        self.preprocessor = None
        self.model = None
        self.labels = []
        
    def load_model(self):
        """Load MobileNet model"""
        if not MOBILENET_AVAILABLE:
            print("MobileNet not available, using fallback")
            return False
            
        try:
            print("Loading MobileNet model...")
            self.preprocessor = AutoImageProcessor.from_pretrained("google/mobilenet_v2_1.0_224")
            self.model = AutoModelForImageClassification.from_pretrained("google/mobilenet_v2_1.0_224")
            self.labels = list(self.model.config.id2label.values())
            print(f"Successfully loaded MobileNet with {len(self.labels)} classes")
            return True
        except Exception as e:
            print(f"Error loading MobileNet: {e}")
            return False
    
    def predict(self, image_bytes):
        """Predict single most confident image class"""
        try:
            # Convert bytes to PIL Image
            pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            
            if self.model is None or self.preprocessor is None:
                return self.create_fallback_prediction()
            
            # Preprocess and predict
            inputs = self.preprocessor(images=pil_image, return_tensors="pt")
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Get the single most confident prediction
            predicted_class_idx = logits.argmax(-1).item()
            raw_class_name = self.model.config.id2label[predicted_class_idx]
            
            # Calculate confidence using softmax
            import torch
            probs = torch.nn.functional.softmax(logits, dim=-1)
            confidence = probs[0][predicted_class_idx].item()
            
            # Normalize the class name to get the most accurate single term
            normalized_class = self.normalize_class_name(raw_class_name)
            
            # Return single prediction with normalized class name
            return {
                'class': normalized_class,
                'confidence': confidence,
                'raw_class': raw_class_name
            }
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return self.create_fallback_prediction()
    
    def normalize_class_name(self, class_name):
        """Normalize ImageNet class names to single most relevant term"""
        # Split by comma and take first term only
        class_name = class_name.split(',')[0].strip()
        
        # Remove underscores and convert to lowercase
        normalized = class_name.replace('_', ' ').lower()
        
        # Map common ImageNet classes to lost & found relevant terms
        # This mapping ensures we get the most accurate single term
        mappings = {
            'cellular telephone': 'cell phone',
            'cellular phone': 'cell phone',
            'mobile phone': 'cell phone',
            'notebook': 'laptop',
            'notebook computer': 'laptop',
            'laptop computer': 'laptop',
            'wrist watch': 'watch',
            'digital watch': 'watch',
            'analog clock': 'clock',
            'sunglasses': 'sunglasses',
            'sunglass': 'sunglasses',
            'backpack': 'backpack',
            'back pack': 'backpack',
            'wallet': 'wallet',
            'billfold': 'wallet',  # Map to wallet
            'notecase': 'wallet',  # Map to wallet
            'pocketbook': 'wallet',  # Map to wallet
            'purse': 'purse',
            'handbag': 'handbag',
            'hand bag': 'handbag',
            'umbrella': 'umbrella',
            'water bottle': 'water bottle',
            'coffee mug': 'coffee mug',
            'computer keyboard': 'keyboard',
            'computer mouse': 'mouse',
            'ipod': 'music player',
            'sunscreen': 'sunscreen',
            'soccer ball': 'soccer ball',
            'basketball': 'basketball',
            'tennis ball': 'tennis ball',
            'paper towel': 'paper',
            'envelope': 'envelope',
            'book jacket': 'book',
            'bookcase': 'bookcase',
            'pill bottle': 'bottle'
        }
        
        return mappings.get(normalized, normalized)
    
    def create_fallback_prediction(self):
        """Fallback prediction when model is not available"""
        return {
            'class': 'personal item',
            'confidence': 0.3,
            'raw_class': 'unknown'
        }


# Initialize the model manager
model_manager = MobileNetModelManager()


# Helper functions
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


def get_primary_object(prediction):
    """Get the primary object from prediction"""
    if not prediction:
        return "unknown", 0.0
    
    return prediction['class'], prediction['confidence']


def extract_features_from_predictions(prediction):
    """Extract features for similarity matching"""
    if not prediction:
        return None
    
    # Create simple feature vector from single prediction
    features = np.zeros(10)
    features[0] = prediction['confidence']
    
    # Hash the class name to a feature index for similarity
    class_hash = hash(prediction['class']) % 9 + 1
    features[class_hash] = prediction['confidence'] * 0.5
    
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


# Initialize model on startup
@app.on_event("startup")
async def startup_event():
    """Load model when the application starts"""
    print("Loading MobileNet image classification model...")
    success = model_manager.load_model()
    if success:
        print(f"Model loaded successfully!")
    else:
        print("Warning: Model loaded with fallbacks")


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
    
    # Use MobileNet for prediction - now returns single most accurate class
    prediction = model_manager.predict(image_data)
    primary_label, confidence = get_primary_object(prediction)
    features = extract_features_from_predictions(prediction)
    
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
        "predicted_label": primary_label,  # Single most confident prediction
        "prediction_details": prediction,  # Full prediction details
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
    
    # Use MobileNet for prediction - now returns single most accurate class
    prediction = model_manager.predict(image_data)
    primary_label, confidence = get_primary_object(prediction)
    features = extract_features_from_predictions(prediction)
    
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
        "predicted_label": primary_label,  # Single primary prediction
        "prediction_details": prediction,  # Complete prediction details
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
    
    # Enhanced text search including predictions
    if query.strip():
        search_filter["$or"] = [
            {"label": {"$regex": query, "$options": "i"}},
            {"predicted_label": {"$regex": query, "$options": "i"}},
            {"prediction_details.class": {"$regex": query, "$options": "i"}},
            {"prediction_details.raw_class": {"$regex": query, "$options": "i"}}
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


# API endpoints for testing
@app.api_route("/predict-label", methods=["POST"])
async def predict_label_api(image: UploadFile = File(...)):
    """API endpoint for testing image classification"""
    try:
        image_data = await image.read()
        prediction = model_manager.predict(image_data)
        primary_label, confidence = get_primary_object(prediction)
        
        return JSONResponse({
            "predicted_label": primary_label,  # Single primary label
            "confidence": confidence,
            "prediction_details": prediction,  # Complete prediction details
            "model_loaded": model_manager.model is not None
        })
    except Exception as e:
        print(f"Error in predict_label_api: {e}")
        return JSONResponse({
            "predicted_label": "error",
            "confidence": 0.0,
            "prediction_details": {},
            "error": str(e)
        })


@app.get("/debug/model-status")
async def model_status():
    """Debug endpoint to check model status"""
    return JSONResponse({
        "model_loaded": model_manager.model is not None,
        "preprocessor_loaded": model_manager.preprocessor is not None,
        "total_classes": len(model_manager.labels),
        "mobilenet_available": MOBILENET_AVAILABLE,
        "sample_classes": model_manager.labels[:20] if model_manager.labels else []
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)