# Vercel serverless function entry point
from api.main import app

# Export the FastAPI app for Vercel
handler = app
