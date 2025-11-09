# 🔍 Lost & Found Web Application

A modern, mobile-responsive web application built with **FastAPI** and **Bootstrap** that helps users report, search, and match lost and found items using AI-powered image recognition and geolocation features.

## ✨ Features

### 🔐 **User Management**
- **JWT Authentication** with secure cookie-based sessions
- User registration and login with phone numbers
- Password hashing with bcrypt

### 📱 **Item Management**
- **Report Lost Items** with photo, location, and description
- **Report Found Items** with optional submission to owners
- **AI-Powered Image Recognition** using MobileNetV3 TensorFlow Lite
- **Automatic Geolocation** detection via browser GPS
- **Manual Location Override** with clickable Google Maps integration

### 🤖 **AI & Machine Learning**
- **MobileNetV3 TFLite** model for object classification
- **Feature Extraction** for visual similarity matching
- **Cosine Similarity** calculations for item matching
- **Confidence Scoring** for predictions
- **ImageNet Labels** support (1000+ object categories)

### 🗺️ **Location Features**
- **GPS Integration** for automatic location detection
- **Distance Calculations** using Haversine formula
- **Radius-based Search** (customizable km range)
- **Google Maps Integration** - click any location to view in maps
- **Location Storage** (coordinates + human-readable names)

### 🔍 **Advanced Search**
- **Text-based Search** across item labels and AI predictions
- **Visual Similarity Search** using image features
- **Geographic Filtering** by distance radius
- **Combined Scoring** (similarity + distance)
- **Smart Sorting** prioritizing best matches

### 📊 **Dashboard & Management**
- **Personal Dashboard** showing all user items
- **Item Categories**: Lost Items, Found Items, Items Submitted to You
- **Resolution Tracking** with mark-as-resolved functionality
- **Contact Information** exchange between users
- **Search Similar** button for each lost item

## 🛠️ Technology Stack

### **Backend**
- **FastAPI** - Modern Python web framework
- **MongoDB** - Document database with Motor async driver
- **TensorFlow Lite** - AI model inference
- **JWT** - JSON Web Tokens for authentication
- **Pillow** - Image processing
- **NumPy** - Numerical computations

### **Frontend**
- **Jinja2** - Server-side templating
- **Bootstrap 5** - Responsive UI framework
- **Font Awesome** - Icons
- **Vanilla JavaScript** - Client-side functionality
- **Geolocation API** - Browser GPS access

### **Storage**
- **MongoDB** - User data and item metadata
- **Local Filesystem** - Image storage
- **Feature Vectors** - Stored in MongoDB for similarity matching

## 🚀 Quick Start

### **Prerequisites**
```bash
# Python 3.8+
python --version

# MongoDB running on localhost:27017
mongod --version

# Required Python packages
pip install fastapi uvicorn motor pymongo python-jose[cryptography] passlib[bcrypt] python-multipart pillow tensorflow numpy
```

### **Installation**

1. **Clone and Setup**
```bash
# Create project directory
mkdir lost-found-app
cd lost-found-app

# Create required directories
mkdir -p static/uploads templates

# Copy main.py and all HTML templates to respective directories
```

2. **Download ML Model Files**
```bash
# You need these two files in the root directory:
# mobilenet_v3.tflite - TensorFlow Lite model file
# labels_mobilenet_v3.txt - ImageNet class labels

# Example labels_mobilenet_v3.txt format:
# n01440764 tench, Tinca tinca
# n01443537 goldfish, Carassius auratus
# n01484850 great white shark, white shark
# ... (1000 lines total)
```

3. **Start Services**
```bash
# Start MongoDB
mongod

# Start the application
python main.py
```

4. **Access Application**
```
http://localhost:8000
```

## 📁 Project Structure

```
lost-found-app/
├── main.py                          # FastAPI backend
├── mobilenet_v3.tflite             # AI model file
├── labels_mobilenet_v3.txt          # ImageNet labels
├── static/
│   └── uploads/                     # User-uploaded images
├── templates/
│   ├── base.html                    # Base template with navbar
│   ├── home.html                    # Homepage with nearby items
│   ├── signup.html                  # User registration
│   ├── login.html                   # User authentication
│   ├── lost_item.html              # Report lost item form
│   ├── found_item.html             # Report found item form
│   ├── dashboard.html              # User dashboard
│   ├── search.html                 # Search form
│   └── search_results.html         # Search results display
└── README.md                       # This file
```

## 🔧 Configuration

### **Environment Variables** (Optional)
```python
# In main.py, you can modify these constants:
SECRET_KEY = "your-secret-key-here"  # Change in production!
ACCESS_TOKEN_EXPIRE_MINUTES = 30
UPLOAD_DIR = "static/uploads"
MODEL_PATH = "mobilenet_v3.tflite"
LABELS_PATH = "labels_mobilenet_v3.txt"
```

### **MongoDB Configuration**
```python
# Default connection string
client = AsyncIOMotorClient("mongodb://localhost:27017")
db = client.lost_found

# Collections created automatically:
# - users: User accounts and authentication
# - items: Lost and found items with metadata
```

## 📱 Usage Guide

### **For Users Who Lost Items:**

1. **Sign Up/Login** with your phone number
2. **Report Lost Item:**
   - Upload clear photo of the lost item
   - AI will suggest what the item is
   - Add your own description
   - Use GPS or manually enter where you lost it
   - Specify when you lost it

3. **Search for Your Item:**
   - Go to Dashboard → Click "Search Similar" 
   - AI will find visually similar found items
   - Results show similarity percentage and distance
   - Contact finders directly via phone

### **For Users Who Found Items:**

1. **Report Found Item:**
   - Upload photo of found item
   - AI suggests item category
   - Add location where you found it
   - Optionally submit directly to owner if you know them

2. **Help Others Find Their Items:**
   - Your found items appear in search results
   - Users can contact you via your phone number

### **Dashboard Features:**

- **Lost Items**: Your reported lost items + search functionality
- **Found Items**: Items you've found and reported
- **Submitted Items**: Items others submitted to you
- **Resolution**: Mark items as resolved when found/returned

## 🤖 AI Model Details

### **MobileNetV3 Integration**
- **Model Type**: TensorFlow Lite optimized for mobile
- **Input Size**: 224x224 RGB images
- **Preprocessing**: Normalized to [-1, 1] range
- **Output**: 1000 ImageNet class probabilities + feature vectors
- **Feature Extraction**: Used for visual similarity matching

### **Similarity Matching**
```python
# Cosine similarity between feature vectors
similarity = dot(features1, features2) / (norm(features1) * norm(features2))

# Results ranked by:
# 1. Visual similarity percentage (if available)
# 2. Geographic distance (km)
# 3. Text relevance
```

### **Supported Object Categories**
The model recognizes 1000+ ImageNet categories including:
- Electronics (phones, laptops, cameras)
- Accessories (bags, wallets, jewelry)
- Clothing items
- Sports equipment
- Household items
- And much more...

## 🗺️ Location Features

### **GPS Integration**
```javascript
// Automatic location detection
navigator.geolocation.getCurrentPosition(callback)

// Distance calculation (Haversine formula)
distance = calculateDistance(lat1, lon1, lat2, lon2)
```

### **Google Maps Integration**
- Click any location text to open in Google Maps
- Shows exact coordinates + location name
- Opens in new tab for easy navigation

## 🔍 API Endpoints

### **Public Routes**
- `GET /` - Homepage with nearby lost items
- `GET /signup` - Registration page
- `GET /login` - Login page
- `POST /signup` - User registration
- `POST /login` - User authentication

### **Protected Routes** (Requires Authentication)
- `GET /dashboard` - User dashboard
- `GET /lost-item` - Report lost item form
- `POST /lost-item` - Submit lost item
- `GET /found-item` - Report found item form
- `POST /found-item` - Submit found item
- `GET /search` - Search form
- `POST /search` - Search results
- `POST /resolve-item/{id}` - Mark item as resolved
- `GET /logout` - User logout

### **API Routes**
- `POST /predict-label` - AI image classification
- `GET /debug/model-status` - Model loading status
- `GET /debug/test-prediction` - Model testing

## 🐛 Troubleshooting

### **Model Not Loading**
```bash
# Check model status
curl http://localhost:8000/debug/model-status

# Common issues:
# 1. Missing model files
# 2. Incorrect file paths
# 3. TensorFlow Lite not installed
# 4. File permissions
```

### **Image Upload Issues**
```bash
# Check upload directory
ls -la static/uploads/

# Ensure directory is writable
chmod 755 static/uploads/
```

### **Location Not Working**
- Enable location services in browser
- Use HTTPS for production (required by browsers)
- Check browser console for geolocation errors

### **Database Connection**
```bash
# Check MongoDB status
mongod --version
mongo --eval "db.runCommand('ping')"

# View collections
mongo lost_found --eval "show collections"
```

## 🔐 Security Considerations

### **Production Deployment**
1. **Change SECRET_KEY** to a secure random string
2. **Use HTTPS** for all traffic
3. **Configure MongoDB authentication**
4. **Set up proper file permissions**
5. **Use environment variables** for sensitive config
6. **Enable MongoDB access control**
7. **Set up proper CORS policies**

### **Data Privacy**
- Phone numbers used for user identification
- GPS coordinates stored for location matching  
- Images stored locally (consider cloud storage for production)
- JWT tokens expire automatically

## 🚀 Deployment

### **Development**
```bash
python main.py
# Runs on http://localhost:8000
```

### **Production** 
```bash
# Using Uvicorn with multiple workers
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4

# Using Docker (create Dockerfile)
# Using cloud services (AWS, GCP, Azure)
# Set up reverse proxy (Nginx)
# Configure SSL certificates
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 📞 Support

For issues and questions:
- Check the troubleshooting section
- Use the debug endpoints
- Review console logs
- Check MongoDB connection

## 🎯 Future Enhancements

- [ ] **Cloud Storage** integration (AWS S3, Google Cloud)
- [ ] **Push Notifications** for item matches
- [ ] **Email Notifications** system
- [ ] **Advanced Filters** (date ranges, item categories)
- [ ] **User Ratings** and feedback system
- [ ] **Admin Dashboard** for moderation
- [ ] **API Rate Limiting** and caching
- [ ] **Mobile App** (React Native/Flutter)
- [ ] **Multiple Image Upload** per item
- [ ] **Social Media Integration** for sharing

---

Built with ❤️ using FastAPI, MongoDB, TensorFlow, and Bootstrap