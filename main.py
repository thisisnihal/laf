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
import tensorflow as tf
from PIL import Image
import io
import math
from typing import Optional, List
import json

# Configuration
SECRET_KEY = "248749f487934"  
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
UPLOAD_DIR = "static/uploads"
MODEL_PATH = "mobilenet_v3.tflite"
LABELS_PATH = "labels_mobilenet_v3.txt"

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

# Load MobileNetV3 model and labels
interpreter = None
labels = []
feature_extractor = None

def load_model():
    global interpreter, labels, feature_extractor
    try:
        # Check if model files exist
        if not os.path.exists(MODEL_PATH):
            print(f"Warning: Model file not found at {MODEL_PATH}")
            return False
        
        if not os.path.exists(LABELS_PATH):
            print(f"Warning: Labels file not found at {LABELS_PATH}")
            return False
            
        # Load TFLite interpreter
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        
        # Load labels
        with open(LABELS_PATH, 'r', encoding='utf-8') as f:
            labels = [line.strip() for line in f.readlines()]
        
        print(f"Model loaded successfully with {len(labels)} labels")
        return True
        
    except Exception as e:
        print(f"Error loading ML model: {e}")
        interpreter = None
        labels = []
        return False

# Initialize model
model_loaded = load_model()

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

def preprocess_image(image_bytes):
    """Preprocess image for MobileNetV3"""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image = image.resize((224, 224))  # MobileNetV3 input size
        image_array = np.array(image).astype(np.float32)
        image_array = np.expand_dims(image_array, axis=0)
        # MobileNetV3 preprocessing: normalize to [-1, 1]
        image_array = (image_array / 127.5) - 1.0
        return image_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def extract_features(image_bytes):
    """Extract feature vector from image using MobileNetV3"""
    if not model_loaded or interpreter is None:
        return None
    
    try:
        processed_image = preprocess_image(image_bytes)
        if processed_image is None:
            return None
            
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        interpreter.set_tensor(input_details[0]['index'], processed_image)
        interpreter.invoke()
        
        # Get feature vector (before final classification layer)
        features = interpreter.get_tensor(output_details[0]['index'])[0]
        return features
        
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def predict_label(image_bytes):
    """Predict label using MobileNetV3"""
    if not model_loaded or interpreter is None:
        print("Model not loaded, returning default")
        return "unknown", 0.0
    
    try:
        processed_image = preprocess_image(image_bytes)
        if processed_image is None:
            return "unknown", 0.0
            
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        interpreter.set_tensor(input_details[0]['index'], processed_image)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Apply softmax to get probabilities
        exp_scores = np.exp(output_data[0] - np.max(output_data[0]))
        probabilities = exp_scores / np.sum(exp_scores)
        
        predicted_index = np.argmax(probabilities)
        confidence = float(probabilities[predicted_index])
        
        if predicted_index < len(labels):
            label = labels[predicted_index]
            # Clean up ImageNet labels (remove synset IDs)
            if ' ' in label:
                label = label.split(' ', 1)[1]  # Remove synset ID
            label = label.split(',')[0]  # Take first synonym
        else:
            label = "unknown"
            
        print(f"Predicted: {label} with confidence {confidence:.3f}")
        return label, confidence
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        return "unknown", 0.0

def calculate_similarity(features1, features2):
    """Calculate cosine similarity between two feature vectors"""
    if features1 is None or features2 is None:
        return 0.0
    
    try:
        # Normalize vectors
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        # Cosine similarity
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
    # Check if user exists
    existing_user = await db.users.find_one({"phone": phone})
    if existing_user:
        return templates.TemplateResponse("signup.html", {
            "request": request, 
            "error": "Phone number already registered"
        })
    
    # Create user
    hashed_password = get_password_hash(password)
    user_data = {
        "name": name,
        "phone": phone,
        "password": hashed_password,
        "created_at": datetime.utcnow()
    }
    
    result = await db.users.insert_one(user_data)
    
    # Create JWT token
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
    # Get auth token from cookie
    token = request.cookies.get("access_token")
    if not token:
        return RedirectResponse(url="/login", status_code=302)
    
    # Remove "Bearer " prefix if present
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
    # Get user from cookie
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
    
    # Extract features and predict label
    features = extract_features(image_data)
    predicted_label, confidence = predict_label(image_data)
    
    # Convert features to list for MongoDB storage
    features_list = features.tolist() if features is not None else []
    
    # Create item
    item_data = {
        "label": label,
        "image_url": f"/static/uploads/{image_filename}",
        "lost_at": lost_location,
        "lost_lat": lost_lat,
        "lost_lon": lost_lon,
        "lost_at_date_time": lost_date_time,
        "submit_at_date_time": datetime.utcnow().isoformat(),
        "image_score": confidence,
        "image_features": features_list,  # Store feature vector
        "lost_by": str(user["_id"]),
        "predicted_label": predicted_label,
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
    # Get user from cookie
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
    
    # Extract features and predict label
    features = extract_features(image_data)
    predicted_label, confidence = predict_label(image_data)
    
    # Convert features to list for MongoDB storage
    features_list = features.tolist() if features is not None else []
    
    # Handle submit_to user
    submit_to_user_id = None
    if submit_to_phone:
        submit_to_user = await db.users.find_one({"phone": submit_to_phone})
        if submit_to_user:
            submit_to_user_id = str(submit_to_user["_id"])
    
    # Create item
    item_data = {
        "label": label,
        "image_url": f"/static/uploads/{image_filename}",
        "found_at": found_location,
        "found_lat": found_lat,
        "found_lon": found_lon,
        "found_at_date_time": found_date_time,
        "submit_at_date_time": datetime.utcnow().isoformat(),
        "image_score": confidence,
        "image_features": features_list,  # Store feature vector
        "found_by": str(user["_id"]),
        "predicted_label": predicted_label,
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
    
    # Get user's items
    lost_items = await db.items.find({"lost_by": user_id}).to_list(100)
    found_items = await db.items.find({"found_by": user_id}).to_list(100)
    submitted_items = await db.items.find({"submit_to": user_id}).to_list(100)
    
    # Get phone numbers for submitted items
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
    
    # Get item details if item_id provided
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
    
    # Get search item for similarity matching
    search_item = None
    search_features = None
    if item_id:
        search_item = await db.items.find_one({"_id": ObjectId(item_id)})
        if search_item and search_item.get("image_features"):
            search_features = np.array(search_item["image_features"])
    
    # Search for found items matching the query
    search_filter = {
        "found_by": {"$exists": True},
        "resolved": {"$ne": True}
    }
    
    # Add text search if query provided
    if query.strip():
        search_filter["$or"] = [
            {"label": {"$regex": query, "$options": "i"}},
            {"predicted_label": {"$regex": query, "$options": "i"}}
        ]
    
    found_items = await db.items.find(search_filter).to_list(200)
    
    # Filter by distance and calculate similarity
    nearby_items = []
    for item in found_items:
        if item.get("found_lat") and item.get("found_lon"):
            distance = calculate_distance(lat, lon, item["found_lat"], item["found_lon"])
            if distance <= radius:
                item["distance"] = round(distance, 2)
                
                # Calculate image similarity if we have search features
                if search_features is not None and item.get("image_features"):
                    item_features = np.array(item["image_features"])
                    similarity = calculate_similarity(search_features, item_features)
                    item["similarity"] = round(similarity * 100, 1)  # Convert to percentage
                else:
                    item["similarity"] = 0.0
                
                # Get finder info
                finder = await db.users.find_one({"_id": ObjectId(item["found_by"])})
                if finder:
                    item["finder_phone"] = finder["phone"]
                    item["finder_name"] = finder["name"]
                nearby_items.append(item)
    
    # Sort by similarity first (if available), then by distance
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
    
    # Update item as resolved
    await db.items.update_one(
        {"_id": ObjectId(item_id)},
        {"$set": {"resolved": True, "resolved_at": datetime.utcnow().isoformat()}}
    )
    
    return JSONResponse({"success": True})

@app.api_route("/predict-label", methods=["POST"])
async def predict_label_api(image: UploadFile = File(...)):
    """API endpoint for getting ML predictions"""
    try:
        image_data = await image.read()
        predicted_label, confidence = predict_label(image_data)
        
        return JSONResponse({
            "predicted_label": predicted_label,
            "confidence": confidence,
            "model_loaded": model_loaded
        })
    except Exception as e:
        print(f"Error in predict_label_api: {e}")
        return JSONResponse({
            "predicted_label": "unknown",
            "confidence": 0.0,
            "model_loaded": False,
            "error": str(e)
        })

@app.get("/debug/model-status")
async def model_status():
    """Debug endpoint to check model status"""
    return JSONResponse({
        "model_loaded": model_loaded,
        "model_path_exists": os.path.exists(MODEL_PATH),
        "labels_path_exists": os.path.exists(LABELS_PATH),
        "labels_count": len(labels),
        "interpreter_loaded": interpreter is not None,
        "model_path": MODEL_PATH,
        "labels_path": LABELS_PATH,
        "sample_labels": labels[:10] if labels else []
    })

@app.get("/debug/test-prediction")
async def test_prediction():
    """Debug endpoint to test model prediction without image"""
    return JSONResponse({
        "message": "Upload an image to /predict-label to test predictions",
        "model_status": {
            "loaded": model_loaded,
            "interpreter": interpreter is not None,
            "labels_available": len(labels)
        }
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)