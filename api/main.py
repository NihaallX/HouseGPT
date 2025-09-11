"""
FastAPI Backend for HouseGPT Web Interface

Simple API server that exposes our working House model as a web service.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import logging
import traceback
import time
from datetime import datetime

# Configure logging for debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

app = FastAPI(title="HouseGPT API", description="Gregory House AI Chatbot API", version="1.0.0")

# Enable CORS for web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model (will be initialized on startup)
house_model = None
style_filter = None
config = None
logger = logging.getLogger(__name__)

# Request/Response models
class ChatMessage(BaseModel):
    message: str
    user_id: Optional[str] = "anonymous"

class ChatResponse(BaseModel):
    response: str
    confidence: float
    sarcasm_level: int
    response_time: float
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    uptime: float

# Store startup time for uptime calculation
startup_time = time.time()

@app.on_event("startup")
async def startup_event():
    """Initialize the House model on server startup."""
    global house_model, style_filter, config
    
    logger.info("🏥 Starting HouseGPT API server...")
    
    try:
        # Add parent directory to path for imports
        import sys
        from pathlib import Path
        parent_dir = Path(__file__).parent.parent
        sys.path.insert(0, str(parent_dir))
        
        # Import here to avoid issues if modules aren't available
        from src.services.house_model import HouseModel
        from src.services.style_filter import StyleFilter
        from src.models.configuration import Configuration
        from src.models.context import ConversationContext
        
        logger.info("📦 Loading configuration...")
        config = Configuration()
        config.model.lora_model_path = "housegpt-lora-large/"
        logger.info("✅ Configuration loaded")
        
        logger.info("🤖 Loading House model...")
        house_model = HouseModel(config, strict_mode=False)
        logger.info("✅ House model loaded successfully")
        
        logger.info("🎭 Loading style filter...")
        style_filter = StyleFilter(config)
        logger.info("✅ Style filter loaded")
        
        logger.info("🎉 HouseGPT API is ready!")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize HouseGPT: {e}")
        logger.error(traceback.format_exc())
        raise

@app.get("/", response_model=dict)
async def root():
    """Root endpoint with basic info."""
    return {
        "message": "HouseGPT API is running",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    uptime = time.time() - startup_time
    return HealthResponse(
        status="healthy" if house_model is not None else "unhealthy",
        model_loaded=house_model is not None,
        uptime=uptime
    )

@app.post("/chat", response_model=ChatResponse)
async def chat_with_house(chat_request: ChatMessage):
    """Main chat endpoint - send a message to House and get a response."""
    start_time = time.time()
    
    logger.info(f"💬 Received chat request from {chat_request.user_id}: {chat_request.message}")
    
    if house_model is None or style_filter is None:
        logger.error("❌ Model not initialized")
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    if not chat_request.message.strip():
        logger.warning("⚠️ Empty message received")
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    try:
        # Import ConversationContext with proper path handling
        import sys
        from pathlib import Path
        parent_dir = Path(__file__).parent.parent
        if str(parent_dir) not in sys.path:
            sys.path.insert(0, str(parent_dir))
        
        from src.models.context import ConversationContext
        
        # Create conversation context
        context = ConversationContext(
            user_query=chat_request.message.strip(),
            conversation_history=[],  # TODO: Implement conversation history
            relevant_quotes=[]  # TODO: Add quote retrieval
        )
        
        logger.info(f"🤔 Generating House response for: '{chat_request.message[:50]}...'")
        
        # Generate House response
        house_response = house_model.generate_response(context)
        
        # Extract response text
        response_text = house_response.text if hasattr(house_response, 'text') else str(house_response)
        
        # Get confidence and sarcasm level
        confidence = getattr(house_response, 'confidence_score', 0.8)
        sarcasm_level = getattr(house_response, 'sarcasm_level', 3)
        
        # Calculate response time
        response_time = time.time() - start_time
        
        logger.info(f"✅ Generated response in {response_time:.2f}s: '{response_text[:50]}...'")
        
        # Return response
        return ChatResponse(
            response=response_text,
            confidence=confidence,
            sarcasm_level=sarcasm_level,
            response_time=response_time,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"❌ Error generating response: {e}")
        logger.error(traceback.format_exc())
        
        # Return fallback response
        fallback_responses = [
            "Everybody lies. What's your point?",
            "People are idiots. This is not news.",
            "Right, because that makes perfect sense... if you're brain-dead.",
            "Fascinating. And by fascinating, I mean completely predictable."
        ]
        
        import random
        fallback_response = random.choice(fallback_responses)
        response_time = time.time() - start_time
        
        logger.info(f"🔄 Using fallback response: '{fallback_response}'")
        
        return ChatResponse(
            response=fallback_response,
            confidence=0.5,
            sarcasm_level=5,
            response_time=response_time,
            timestamp=datetime.now().isoformat()
        )

@app.get("/test", response_model=dict)
async def test_endpoint():
    """Test endpoint to verify the model is working."""
    if house_model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    try:
        # Test with a simple message with proper path handling
        import sys
        from pathlib import Path
        parent_dir = Path(__file__).parent.parent
        if str(parent_dir) not in sys.path:
            sys.path.insert(0, str(parent_dir))
        
        from src.models.context import ConversationContext
        
        context = ConversationContext(
            user_query="Hello, how are you?",
            conversation_history=[],
            relevant_quotes=[]
        )
        
        response = house_model.generate_response(context)
        response_text = response.text if hasattr(response, 'text') else str(response)
        
        return {
            "status": "success",
            "test_query": "Hello, how are you?",
            "test_response": response_text,
            "model_loaded": True
        }
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "model_loaded": False
        }

if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Starting HouseGPT API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
