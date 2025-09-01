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
SECRET_KEY = "your-secret-key-here"  # Change this in production
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
try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    with open(LABELS_PATH, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
except Exception as e:
    print(f"Warning: Could not load ML model: {e}")
    interpreter = None
    labels = []

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
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = image.resize((224, 224))
    image_array = np.array(image).astype(np.float32)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = image_array / 255.0  # Normalize
    return image_array

def predict_label(image_bytes):
    """Predict label using MobileNetV3"""
    if interpreter is None:
        return "unknown", 0.0
    
    try:
        processed_image = preprocess_image(image_bytes)
        interpreter.set_tensor(input_details[0]['index'], processed_image)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        predicted_index = np.argmax(output_data[0])
        confidence = float(output_data[0][predicted_index])
        
        if predicted_index < len(labels):
            label = labels[predicted_index]
        else:
            label = "unknown"
            
        return label, confidence
    except Exception as e:
        print(f"Error in prediction: {e}")
        return "unknown", 0.0

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
    
    # Predict label
    predicted_label, confidence = predict_label(image_data)
    
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
    
    # Predict label
    predicted_label, confidence = predict_label(image_data)
    
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
    lon: float = Form(...)
):
    token = request.cookies.get("access_token")
    if not token:
        return RedirectResponse(url="/login", status_code=302)
    
    # Search for found items matching the query
    found_items = await db.items.find({
        "found_by": {"$exists": True},
        "resolved": {"$ne": True},
        "$or": [
            {"label": {"$regex": query, "$options": "i"}},
            {"predicted_label": {"$regex": query, "$options": "i"}}
        ]
    }).to_list(100)
    
    # Filter by distance
    nearby_items = []
    for item in found_items:
        if item.get("found_lat") and item.get("found_lon"):
            distance = calculate_distance(lat, lon, item["found_lat"], item["found_lon"])
            if distance <= radius:
                item["distance"] = round(distance, 2)
                # Get finder info
                finder = await db.users.find_one({"_id": ObjectId(item["found_by"])})
                if finder:
                    item["finder_phone"] = finder["phone"]
                    item["finder_name"] = finder["name"]
                nearby_items.append(item)
    
    nearby_items.sort(key=lambda x: x.get("distance", float('inf')))
    
    return templates.TemplateResponse("search_results.html", {
        "request": request,
        "query": query,
        "radius": radius,
        "items": nearby_items
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
    image_data = await image.read()
    predicted_label, confidence = predict_label(image_data)
    return JSONResponse({
        "predicted_label": predicted_label,
        "confidence": confidence
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)