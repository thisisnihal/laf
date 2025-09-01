# Backend – FastAPI

Requirements
- Python 3.12+
- MongoDB running locally (or a remote URI)

Setup
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp env.example .env
```
Edit `.env` as needed (Mongo URI, JWT secret, DO Spaces creds).

Run
```
python main.py
```
- Base URL: http://localhost:8000
- Health: /health
- API Prefix: /api/v1

Endpoints (overview)
- Auth: /api/v1/auth/register, /api/v1/auth/login, /api/v1/auth/me, /api/v1/auth/refresh
- Items: /api/v1/items (CRUD), /api/v1/items/recent, /api/v1/items/{id}
- Search: /api/v1/search?keyword=&category=&latitude=&longitude=&radius_km=
- Upload: /api/v1/upload (returns public URL)

MongoDB Schema
```
{
  _id: ObjectId,
  title: String,
  category: String,
  description: String,
  location: { type: 'Point', coordinates: [lon, lat] },
  image_url: String,
  user_id: ObjectId,
  created_at: Date,
  updated_at: Date,
  status: 'lost' | 'found' | 'resolved'
}
```

Geospatial
- Ensure a 2dsphere index on location:
```
db.items.createIndex({ location: "2dsphere" })
```

DigitalOcean Spaces
- Set do_spaces_key, do_spaces_secret, do_spaces_endpoint, do_spaces_bucket, do_spaces_region in .env
- Backend uploads files and returns a public URL

Deployment
- Use gunicorn/uvicorn workers or DO App Platform
- Provide env vars (Mongo URI, JWT, DO Spaces)
