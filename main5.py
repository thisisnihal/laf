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
from typing import Optional, List, Dict, Any
import json
import asyncio
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    SECRET_KEY = "248749f487934"
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 30
    UPLOAD_DIR = "static/uploads"
    MONGODB_URL = "mongodb://localhost:27017"
    DATABASE_NAME = "lost_found"
    
    # AI Model Configuration - INTERCHANGEABLE
    MAIN_MODEL = "gemini"  # Change to "mobilenet" to swap roles
    FALLBACK_MODEL = "mobilenet"
    CONFIDENCE_THRESHOLD = 0.7
    
    # Gemini Configuration
    GEMINI_API_KEY = os.getenv("GEMINI_API") 
    
    # Feature Vector Dimensions
    FEATURE_DIMENSIONS = 10

# Ensure upload directory exists
os.makedirs(Config.UPLOAD_DIR, exist_ok=True)

# =============================================================================
# AI MODEL INTERFACES
# =============================================================================

class BaseAIModel:
    """Base interface for all AI models"""
    
    def __init__(self):
        self.available = False
        self.model_name = "base"
    
    def load_model(self) -> bool:
        """Load the model, return success status"""
        raise NotImplementedError
    
    async def predict(self, image_bytes: bytes, user_label: str = "") -> Dict[str, Any]:
        """Predict item category from image"""
        raise NotImplementedError
    
    def extract_features(self, prediction: Dict[str, Any]) -> List[float]:
        """Extract feature vector for similarity matching"""
        raise NotImplementedError
    
    def create_fallback_prediction(self) -> Dict[str, Any]:
        """Create fallback prediction when model fails"""
        return {
            'class': 'personal item',
            'confidence': 0.3,
            'model': self.model_name,
            'fallback': True
        }


class MobileNetModel(BaseAIModel):
    """MobileNet V2 implementation"""
    
    def __init__(self):
        super().__init__()
        self.model_name = "mobilenet_v2"
        self.preprocessor = None
        self.model = None
        self.labels = []
        self.enhanced_mappings = self._get_enhanced_mappings()
    
    def _get_enhanced_mappings(self) -> Dict[str, str]:
        """Comprehensive mapping for lost & found items"""
        return {
            # Wallet family
            'wallet': 'wallet',
            'billfold': 'wallet',
            'notecase': 'wallet', 
            'pocketbook': 'wallet',
            'purse': 'purse',
            
            # Phone family
            'cellular telephone': 'phone',
            'cellular phone': 'phone',
            'mobile phone': 'phone',
            'smartphone': 'phone',
            
            # Computer family
            'notebook': 'laptop',
            'notebook computer': 'laptop',
            'laptop computer': 'laptop',
            
            # Watch family
            'wrist watch': 'watch',
            'digital watch': 'watch',
            'analog watch': 'watch',
            
            # Bag family
            'backpack': 'backpack',
            'back pack': 'backpack',
            'handbag': 'handbag',
            'hand bag': 'handbag',
            
            # Electronics
            'computer keyboard': 'keyboard',
            'computer mouse': 'mouse',
            'ipod': 'music player',
            'earphone': 'earphones',
            
            # Personal items
            'sunglasses': 'sunglasses',
            'eyeglasses': 'glasses',
            'umbrella': 'umbrella',
            'water bottle': 'water bottle',
            'coffee mug': 'coffee mug',
            
            # Sports equipment
            'soccer ball': 'soccer ball',
            'basketball': 'basketball',
            'tennis ball': 'tennis ball',
            
            # Documents
            'envelope': 'envelope',
            'book': 'book',
            'notebook': 'notebook',
        }
    
    def load_model(self) -> bool:
        """Load MobileNet model"""
        try:
            from transformers import AutoImageProcessor, AutoModelForImageClassification
            print("Loading MobileNet model...")
            self.preprocessor = AutoImageProcessor.from_pretrained("google/mobilenet_v2_1.0_224")
            self.model = AutoModelForImageClassification.from_pretrained("google/mobilenet_v2_1.0_224")
            self.labels = list(self.model.config.id2label.values())
            self.available = True
            print(f"MobileNet loaded with {len(self.labels)} classes")
            return True
        except ImportError:
            print("MobileNet not available. Install: pip install transformers torch")
            return False
        except Exception as e:
            print(f"Error loading MobileNet: {e}")
            return False
    
    async def predict(self, image_bytes: bytes, user_label: str = "") -> Dict[str, Any]:
        """Predict using MobileNet"""
        try:
            pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            
            if not self.available:
                return self.create_fallback_prediction()
            
            # MobileNet processing
            inputs = self.preprocessor(images=pil_image, return_tensors="pt")
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Get top prediction
            predicted_class_idx = logits.argmax(-1).item()
            raw_class_name = self.model.config.id2label[predicted_class_idx]
            
            # Calculate confidence
            import torch
            probs = torch.nn.functional.softmax(logits, dim=-1)
            confidence = probs[0][predicted_class_idx].item()
            
            # Normalize class name
            normalized_class = self._normalize_class_name(raw_class_name)
            
            return {
                'class': normalized_class,
                'confidence': confidence,
                'raw_class': raw_class_name,
                'model': self.model_name,
                'features': self.extract_features_from_prediction(normalized_class, confidence)
            }
            
        except Exception as e:
            print(f"MobileNet prediction error: {e}")
            return self.create_fallback_prediction()
    
    def _normalize_class_name(self, class_name: str) -> str:
        """Normalize class name using enhanced mappings"""
        class_name = class_name.split(',')[0].strip().lower().replace('_', ' ')
        return self.enhanced_mappings.get(class_name, class_name)
    
    def extract_features(self, prediction: Dict[str, Any]) -> List[float]:
        """Extract feature vector from prediction"""
        features = prediction.get('features')
        if features is not None:
            if hasattr(features, 'tolist'):
                return features.tolist()
            return features
        return self.extract_features_from_prediction(
            prediction.get('class', 'unknown'), 
            prediction.get('confidence', 0.0)
        )
    
    def extract_features_from_prediction(self, class_name: str, confidence: float) -> List[float]:
        """Create feature vector from class name and confidence"""
        features = np.zeros(Config.FEATURE_DIMENSIONS)
        features[0] = confidence
        
        # Hash class name to feature positions
        class_hash = hash(class_name) % (Config.FEATURE_DIMENSIONS - 1) + 1
        features[class_hash] = confidence * 0.5
        
        return features.tolist()


class GeminiModel(BaseAIModel):
    """Gemini Pro Vision implementation"""
    
    def __init__(self, api_key: str):
        super().__init__()
        self.model_name = "gemini-2.5-flash"
        self.api_key = api_key
        self.model = None
    
    def load_model(self) -> bool:
        """Load Gemini model"""
        try:
            import google.generativeai as genai
            if not self.api_key:
                print("Gemini API key not configured")
                return False
            
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-2.5-flash')
            self.available = True
            print("Gemini Pro Vision loaded successfully")
            return True
        except ImportError:
            print("Gemini not available. Install: pip install google-generativeai")
            return False
        except Exception as e:
            print(f"Error loading Gemini: {e}")
            return False
    
    async def predict(self, image_bytes: bytes, user_label: str = "") -> Dict[str, Any]:
        """Predict using Gemini Pro Vision"""
        try:
            pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            
            if not self.available:
                return self.create_fallback_prediction()
            
            # Enhanced prompt for lost & found context
            prompt = """
            Analyze this image for a lost and found application. 
            What is the main item in this image? 
            Return ONLY a single, specific item category name (e.g., "wallet", "phone", "keys", "backpack").
            Be specific but concise - just the item name.
            
            Important: Return ONLY the category name, no explanations, no punctuation, just the word.
            """
            
            if user_label:
                prompt += f"\nUser description: {user_label}"
            
            response = await self.model.generate_content_async([prompt, pil_image])
            category = response.text.strip().lower().replace('.', '')
            
            # Clean and validate response
            category = self._clean_category_name(category)
            
            # Gemini doesn't provide confidence scores, so we use a high default
            confidence = 0.85
            
            return {
                'class': category,
                'confidence': confidence,
                'raw_response': response.text,
                'model': self.model_name,
                'features': self.extract_features_from_prediction(category, confidence)
            }
            
        except Exception as e:
            print(f"Gemini prediction error: {e}")
            return self.create_fallback_prediction()
    
    def _clean_category_name(self, category: str) -> str:
        """Clean and standardize category names from Gemini"""
        # Remove any extra text and keep only the first word/phrase
        category = category.split('\n')[0].split('.')[0].strip()
        
        # Common normalizations
        mappings = {
            'cell phone': 'phone',
            'mobile phone': 'phone',
            'smartphone': 'phone',
            'purse': 'handbag',
            'billfold': 'wallet',
            'notecase': 'wallet',
            'pocketbook': 'wallet',
            'laptop computer': 'laptop',
            'notebook computer': 'laptop',
            'wristwatch': 'watch',
            'digital watch': 'watch',
            'back pack': 'backpack'
        }
        
        return mappings.get(category, category)
    
    def extract_features(self, prediction: Dict[str, Any]) -> List[float]:
        """Extract feature vector from prediction"""
        features = prediction.get('features')
        if features is not None:
            if hasattr(features, 'tolist'):
                return features.tolist()
            return features
        return self.extract_features_from_prediction(
            prediction.get('class', 'unknown'), 
            prediction.get('confidence', 0.0)
        )
    
    def extract_features_from_prediction(self, class_name: str, confidence: float) -> List[float]:
        """Create feature vector from class name and confidence"""
        features = np.zeros(Config.FEATURE_DIMENSIONS)
        features[0] = confidence
        
        # Hash class name to feature positions
        class_hash = hash(class_name) % (Config.FEATURE_DIMENSIONS - 1) + 1
        features[class_hash] = confidence * 0.5
        
        return features.tolist()


class HybridModelManager:
    """Orchestrator that manages main and fallback models"""
    
    def __init__(self, main_model: str, fallback_model: str, gemini_api_key: str = None):
        self.main_model_name = main_model
        self.fallback_model_name = fallback_model
        self.confidence_threshold = Config.CONFIDENCE_THRESHOLD
        
        # Initialize models
        self.main_model = self._create_model(main_model, gemini_api_key)
        self.fallback_model = self._create_model(fallback_model, gemini_api_key)
        
        print(f"Model Configuration: {main_model.upper()} (Main) -> {fallback_model.upper()} (Fallback)")
    
    def _create_model(self, model_type: str, gemini_api_key: str) -> BaseAIModel:
        """Create model instance based on type"""
        if model_type == "gemini":
            return GeminiModel(gemini_api_key)
        elif model_type == "mobilenet":
            return MobileNetModel()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def load_models(self) -> bool:
        """Load both main and fallback models"""
        main_loaded = self.main_model.load_model()
        fallback_loaded = self.fallback_model.load_model()
        
        print(f"Main model ({self.main_model_name}): {'Loaded' if main_loaded else 'Failed'}")
        print(f"Fallback model ({self.fallback_model_name}): {'Loaded' if fallback_loaded else 'Failed'}")
        
        # System can work if at least one model is loaded
        return main_loaded or fallback_loaded
    
    async def predict(self, image_bytes: bytes, user_label: str = "") -> Dict[str, Any]:
        """Predict using main model with fallback support"""
        
        # Step 1: Try main model
        if self.main_model.available:
            try:
                main_prediction = await self.main_model.predict(image_bytes, user_label)
                
                # Step 2: Check if we need fallback (low confidence)
                if main_prediction.get('confidence', 0) >= self.confidence_threshold:
                    print(f"Main model confident: {main_prediction['class']} ({main_prediction['confidence']:.2f})")
                    return {
                        **main_prediction,
                        'primary_model': self.main_model_name,
                        'fallback_used': False
                    }
                else:
                    print(f"Main model low confidence: {main_prediction['confidence']:.2f}, trying fallback...")
            except Exception as e:
                print(f"Main model failed: {e}, trying fallback...")
                main_prediction = None
        else:
            print("Main model not available, trying fallback...")
            main_prediction = None
        
        # Step 3: Use fallback model
        if self.fallback_model.available:
            try:
                fallback_prediction = await self.fallback_model.predict(image_bytes, user_label)
                print(f"Fallback model result: {fallback_prediction['class']} ({fallback_prediction['confidence']:.2f})")
                
                # Merge results if we have main prediction
                if main_prediction:
                    return self._merge_predictions(main_prediction, fallback_prediction)
                else:
                    return {
                        **fallback_prediction,
                        'primary_model': self.fallback_model_name,
                        'fallback_used': True,
                        'main_model_failed': True
                    }
                    
            except Exception as e:
                print(f"Fallback model also failed: {e}")
        
        # Step 4: Both models failed, use fallback prediction
        print("All models failed, using ultimate fallback")
        ultimate_fallback = self.main_model.create_fallback_prediction()
        return {
            **ultimate_fallback,
            'primary_model': 'none',
            'fallback_used': True,
            'all_models_failed': True
        }
    
    def _merge_predictions(self, main_pred: Dict, fallback_pred: Dict) -> Dict:
        """Merge predictions from main and fallback models"""
        # Prefer fallback if main had low confidence
        merged_class = fallback_pred['class']
        merged_confidence = max(main_pred['confidence'], fallback_pred['confidence'])
        
        return {
            'class': merged_class,
            'confidence': merged_confidence,
            'primary_model': self.main_model_name,
            'fallback_used': True,
            'main_model_prediction': main_pred,
            'fallback_model_prediction': fallback_pred,
            'features': fallback_pred.get('features', main_pred.get('features'))
        }
    
    def extract_features(self, prediction: Dict[str, Any]) -> List[float]:
        """Extract features from prediction"""
        if 'features' in prediction:
            features = prediction['features']
            if hasattr(features, 'tolist'):
                return features.tolist()
            return features
        
        # Fallback feature extraction
        features = np.zeros(Config.FEATURE_DIMENSIONS)
        features[0] = prediction.get('confidence', 0.0)
        
        class_name = prediction.get('class', 'unknown')
        class_hash = hash(class_name) % (Config.FEATURE_DIMENSIONS - 1) + 1
        features[class_hash] = prediction.get('confidence', 0.0) * 0.5
        
        return features.tolist()


# =============================================================================
# FASTAPI APPLICATION SETUP
# =============================================================================

app = FastAPI(title="Lost & Found App")

# Initialize AI Model Manager - INTERCHANGEABLE MODELS
model_manager = HybridModelManager(
    main_model=Config.MAIN_MODEL,
    fallback_model=Config.FALLBACK_MODEL,
    gemini_api_key=Config.GEMINI_API_KEY
)

# MongoDB setup
client = AsyncIOMotorClient(Config.MONGODB_URL)
db = client[Config.DATABASE_NAME]

# Static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Security
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer(auto_error=False)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, Config.SECRET_KEY, algorithm=Config.ALGORITHM)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials is None:
        return None
    
    try:
        payload = jwt.decode(credentials.credentials, Config.SECRET_KEY, algorithms=[Config.ALGORITHM])
        phone: str = payload.get("sub")
        if phone is None:
            return None
    except JWTError:
        return None
    
    return await db.users.find_one({"phone": phone})

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
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

def calculate_similarity(features1: np.ndarray, features2: np.ndarray) -> float:
    """Calculate cosine similarity between feature vectors"""
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


# =============================================================================
# APPLICATION STARTUP
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    print("Starting Lost & Found Application...")
    print("Loading AI Models...")
    
    success = model_manager.load_models()
    if success:
        print("AI Models loaded successfully!")
    else:
        print("Some models failed to load, but application will continue")
    
    print(f"Configuration: {Config.MAIN_MODEL.upper()} -> {Config.FALLBACK_MODEL.upper()}")
    print("Application startup complete!")


# =============================================================================
# API ROUTES
# =============================================================================

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
    
    return templates.TemplateResponse("home.html", {
        "request": request, 
        "user": user, 
        "items": items,
        "main_model": Config.MAIN_MODEL.upper()
    })

@app.get("/signup", response_class=HTMLResponse)
async def signup_page(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})

@app.post("/signup")
async def signup(
    request: Request, 
    name: str = Form(...), 
    phone: str = Form(...), 
    password: str = Form(...)
):
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
    access_token_expires = timedelta(minutes=Config.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": phone}, expires_delta=access_token_expires)
    
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
    
    access_token_expires = timedelta(minutes=Config.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": phone}, expires_delta=access_token_expires)
    
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
        payload = jwt.decode(token, Config.SECRET_KEY, algorithms=[Config.ALGORITHM])
        phone = payload.get("sub")
        user = await db.users.find_one({"phone": phone})
        if not user:
            return RedirectResponse(url="/login", status_code=302)
    except JWTError:
        return RedirectResponse(url="/login", status_code=302)
    
    return templates.TemplateResponse("lost_item.html", {
        "request": request, 
        "user": user,
        "main_model": Config.MAIN_MODEL.upper()
    })

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
        payload = jwt.decode(token, Config.SECRET_KEY, algorithms=[Config.ALGORITHM])
        phone = payload.get("sub")
        user = await db.users.find_one({"phone": phone})
        if not user:
            return RedirectResponse(url="/login", status_code=302)
    except JWTError:
        return RedirectResponse(url="/login", status_code=302)
    
    # Save image
    image_data = await image.read()
    image_filename = f"{uuid.uuid4()}.jpg"
    image_path = os.path.join(Config.UPLOAD_DIR, image_filename)
    
    with open(image_path, "wb") as f:
        f.write(image_data)
    
    # AI Prediction
    prediction = await model_manager.predict(image_data, label)
    features = model_manager.extract_features(prediction)
    
    item_data = {
        "label": label,
        "image_url": f"/static/uploads/{image_filename}",
        "lost_at": lost_location,
        "lost_lat": lost_lat,
        "lost_lon": lost_lon,
        "lost_at_date_time": lost_date_time,
        "submit_at_date_time": datetime.utcnow().isoformat(),
        "image_score": prediction['confidence'],
        "image_features": features,
        "lost_by": str(user["_id"]),
        "predicted_label": prediction['class'],
        "prediction_details": prediction,
        "primary_model_used": prediction.get('primary_model', 'unknown'),
        "fallback_used": prediction.get('fallback_used', False),
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
        payload = jwt.decode(token, Config.SECRET_KEY, algorithms=[Config.ALGORITHM])
        phone = payload.get("sub")
        user = await db.users.find_one({"phone": phone})
        if not user:
            return RedirectResponse(url="/login", status_code=302)
    except JWTError:
        return RedirectResponse(url="/login", status_code=302)
    
    return templates.TemplateResponse("found_item.html", {
        "request": request, 
        "user": user,
        "main_model": Config.MAIN_MODEL.upper()
    })

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
        payload = jwt.decode(token, Config.SECRET_KEY, algorithms=[Config.ALGORITHM])
        phone = payload.get("sub")
        user = await db.users.find_one({"phone": phone})
        if not user:
            return RedirectResponse(url="/login", status_code=302)
    except JWTError:
        return RedirectResponse(url="/login", status_code=302)
    
    # Save image
    image_data = await image.read()
    image_filename = f"{uuid.uuid4()}.jpg"
    image_path = os.path.join(Config.UPLOAD_DIR, image_filename)
    
    with open(image_path, "wb") as f:
        f.write(image_data)
    
    # AI Prediction
    prediction = await model_manager.predict(image_data, label)
    features = model_manager.extract_features(prediction)
    
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
        "image_score": prediction['confidence'],
        "image_features": features,
        "found_by": str(user["_id"]),
        "predicted_label": prediction['class'],
        "prediction_details": prediction,
        "primary_model_used": prediction.get('primary_model', 'unknown'),
        "fallback_used": prediction.get('fallback_used', False),
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
        payload = jwt.decode(token, Config.SECRET_KEY, algorithms=[Config.ALGORITHM])
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
        "submitted_items": submitted_items,
        "main_model": Config.MAIN_MODEL.upper()
    })

@app.get("/search", response_class=HTMLResponse)
async def search_page(request: Request, item_id: Optional[str] = None):
    token = request.cookies.get("access_token")
    if not token:
        return RedirectResponse(url="/login", status_code=302)
    
    if token.startswith("Bearer "):
        token = token[7:]
    
    try:
        payload = jwt.decode(token, Config.SECRET_KEY, algorithms=[Config.ALGORITHM])
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
        "search_item": search_item,
        "main_model": Config.MAIN_MODEL.upper()
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
    
    # Enhanced text search
    if query.strip():
        search_filter["$or"] = [
            {"label": {"$regex": query, "$options": "i"}},
            {"predicted_label": {"$regex": query, "$options": "i"}},
            {"prediction_details.class": {"$regex": query, "$options": "i"}}
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
        "has_similarity": search_features is not None,
        "main_model": Config.MAIN_MODEL.upper()
    })

@app.post("/resolve-item/{item_id}")
async def resolve_item(item_id: str, request: Request):
    token = request.cookies.get("access_token")
    if not token:
        return JSONResponse({"success": False, "error": "Not authenticated"})
    
    if token.startswith("Bearer "):
        token = token[7:]
    
    try:
        payload = jwt.decode(token, Config.SECRET_KEY, algorithms=[Config.ALGORITHM])
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

# =============================================================================
# AI API ENDPOINTS
# =============================================================================

@app.api_route("/predict-label", methods=["POST"])
async def predict_label_api(image: UploadFile = File(...), label: str = Form("")):
    """API endpoint for testing image classification"""
    try:
        image_data = await image.read()
        prediction = await model_manager.predict(image_data, label)
        
        def convert_numpy_types(obj):
            """Recursively convert numpy types to Python native types for JSON serialization"""
            if isinstance(obj, (np.ndarray, np.generic)):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        # Convert all numpy types in the prediction
        serializable_prediction = convert_numpy_types(prediction)
        
        return JSONResponse({
            "predicted_label": prediction['class'],
            "confidence": float(prediction['confidence']),
            "primary_model": prediction.get('primary_model', 'unknown'),
            "fallback_used": bool(prediction.get('fallback_used', False)),
            "prediction_details": serializable_prediction,
            "main_model_config": Config.MAIN_MODEL,
            "fallback_model_config": Config.FALLBACK_MODEL
        })
        
    except Exception as e:
        print(f"Error in predict_label_api: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse({
            "predicted_label": "error",
            "confidence": 0.0,
            "error": str(e)
        }, status_code=500)

@app.get("/debug/model-status")
async def model_status():
    """Debug endpoint to check model status"""
    return JSONResponse({
        "main_model": Config.MAIN_MODEL,
        "fallback_model": Config.FALLBACK_MODEL,
        "confidence_threshold": Config.CONFIDENCE_THRESHOLD,
        "main_model_available": model_manager.main_model.available,
        "fallback_model_available": model_manager.fallback_model.available,
        "main_model_name": model_manager.main_model.model_name,
        "fallback_model_name": model_manager.fallback_model.model_name
    })

@app.post("/switch-models")
async def switch_models(main_model: str = Form(...), fallback_model: str = Form(...)):
    """Endpoint to dynamically switch models (for testing)"""
    global model_manager
    
    if main_model not in ["gemini", "mobilenet"] or fallback_model not in ["gemini", "mobilenet"]:
        return JSONResponse({"error": "Invalid model types. Use 'gemini' or 'mobilenet'"}, status_code=400)
    
    if main_model == fallback_model:
        return JSONResponse({"error": "Main and fallback models cannot be the same"}, status_code=400)
    
    # Update configuration
    Config.MAIN_MODEL = main_model
    Config.FALLBACK_MODEL = fallback_model
    
    # Reinitialize model manager
    model_manager = HybridModelManager(
        main_model=Config.MAIN_MODEL,
        fallback_model=Config.FALLBACK_MODEL,
        gemini_api_key=Config.GEMINI_API_KEY
    )
    
    success = model_manager.load_models()
    
    return JSONResponse({
        "success": success,
        "new_main_model": Config.MAIN_MODEL,
        "new_fallback_model": Config.FALLBACK_MODEL,
        "models_loaded": success
    })

# =============================================================================
# APPLICATION ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    print(f"Starting with: {Config.MAIN_MODEL.upper()} (Main) -> {Config.FALLBACK_MODEL.upper()} (Fallback)")
    uvicorn.run(app, host="0.0.0.0", port=8002)