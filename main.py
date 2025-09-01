from fastapi import FastAPI, Request, Form, UploadFile, File, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from passlib.hash import bcrypt
import jwt, datetime, os, requests, json
from pymongo import MongoClient
from bson import ObjectId
from PIL import Image
import numpy as np
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity

# ----------------- CONFIG -----------------
SECRET_KEY = "supersecret"
MONGO_URL = "mongodb://localhost:27017"
DB_NAME = "lostfound"

app = FastAPI()
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
templates = Jinja2Templates(directory="templates")

client = MongoClient(MONGO_URL)
db = client[DB_NAME]

# ----------------- ML MODEL -----------------
interpreter = tf.lite.Interpreter(model_path="mobilenet_v3.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_image(img: Image.Image):
    img = img.resize((224, 224))  # model input size
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

# Download ImageNet labels (only once)
url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
if not os.path.exists("imagenet_labels.txt"):
    data = requests.get(url).json()
    with open("imagenet_labels.txt", "w") as f:
        f.write("background\n")  # index 0
        for k in range(len(data)):
            f.write(data[str(k)][1] + "\n")

# Load labels into list
with open("imagenet_labels.txt") as f:
    labels = [line.strip() for line in f.readlines()]

def get_embedding_and_label(img: Image.Image):
    arr = preprocess_image(img)
    interpreter.set_tensor(input_details[0]['index'], arr)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])[0]

    label_id = int(np.argmax(preds))
    confidence = float(np.max(preds))
    label_name = labels[label_id] if label_id < len(labels) else str(label_id)

    return label_name, confidence, preds


def predict_image(image: Image.Image):
    # resize to model’s input shape
    image = Image.open(image)
    img = image.resize((224, 224))
    target_size = input_details[0]['shape'][1:3]
    img = image.resize(target_size)
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])
    return preds.tolist()


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# ----------------- AUTH -----------------
def create_token(user_id: str):
    payload = {"sub": str(user_id), "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=2)}
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

def decode_token(token: str):
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    except:
        return None

# ----------------- ROUTES -----------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    items = list(db.items.find().sort("submit_at_date_time", -1))
    return templates.TemplateResponse("home.html", {"request": request, "items": items})

@app.get("/signup")
async def signup_page(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})

@app.post("/signup")
async def signup(name: str = Form(...), phone: str = Form(...), password: str = Form(...)):
    hashed = bcrypt.hash(password)
    db.users.insert_one({"name": name, "phone": phone, "password": hashed})
    return RedirectResponse("/login", status_code=302)

@app.get("/login")
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
async def login(phone: str = Form(...), password: str = Form(...)):
    user = db.users.find_one({"phone": phone})
    if not user or not bcrypt.verify(password, user["password"]):
        return RedirectResponse("/login", status_code=302)
    token = create_token(user["_id"])
    response = RedirectResponse("/", status_code=302)
    response.set_cookie("token", token)
    return response

@app.get("/lost")
async def lost_page(request: Request):
    return templates.TemplateResponse("lost.html", {"request": request})

import math

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate great-circle distance between two coordinates (in km).
    """
    R = 6371  # Earth radius in km
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

@app.post("/lost")
async def create_lost_item(
    label: str = Form(...),
    file: UploadFile = File(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
):
    # Save file
    file_path = f"uploads/{file.filename}"
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    # Run through TFLite model
    label_idx, confidence, embedding = predict_image(file_path)
    predicted_label = labels[label_idx]

    # Store in MongoDB
    item = {
        "label": label or predicted_label,
        "predicted_label": predicted_label,
        "confidence": confidence,
        "image_url": f"/{file_path}",
        "status": "lost",
        "latitude": latitude,
        "longitude": longitude,
        "embedding": embedding.tolist(),
    }
    result = await db["items"].insert_one(item)
    return {"id": str(result.inserted_id), "prediction": item}


@app.get("/dashboard")
async def dashboard(request: Request):
    token = request.cookies.get("token")
    user_data = decode_token(token)
    if not user_data:
        return RedirectResponse("/login", status_code=302)

    items = list(db.items.find({"lost_by": user_data["sub"]}))
    return templates.TemplateResponse("dashboard.html", {"request": request, "items": items})





@app.get("/match/{item_id}")
async def match_item(item_id: str, radius: float = 2.0):
    """
    Match a found item to lost items within given radius (km).
    """
    item = await db["items"].find_one({"_id": item_id})
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")

    item_vec = np.array(item["embedding"]).reshape(1, -1)
    results = []

    async for other in db["items"].find({"status": "lost", "_id": {"$ne": item_id}}):
        distance = haversine(
            item["longitude"], item["latitude"], other["longitude"], other["latitude"]
        )
        if distance > radius:
            continue

        other_vec = np.array(other["embedding"]).reshape(1, -1)
        score = cosine_similarity(item_vec, other_vec)[0][0]

        if score > 0.7:
            other["similarity_score"] = float(score)
            other["distance_km"] = round(distance, 2)
            results.append(other)

    return {"matches": results}
