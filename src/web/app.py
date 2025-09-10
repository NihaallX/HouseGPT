"""
HouseGPT Web Interface

FastAPI-based web server providing HTTP API and web interface for HouseGPT.
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, List, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn

# HouseGPT imports - loaded lazily
house_model = None
style_filter = None
rag_store = None
voice_output = None
conversation_logger = None
config = None


# Request/Response models
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    voice_enabled: bool = False


class ChatResponse(BaseModel):
    response: str
    session_id: str
    processing_time_ms: float
    style_score: float
    confidence: float
    rag_quotes_used: int
    voice_available: bool = False


class HealthResponse(BaseModel):
    status: str
    services: Dict[str, bool]
    uptime_seconds: float
    version: str


class StatsResponse(BaseModel):
    total_conversations: int
    successful_conversations: int
    failed_conversations: int
    average_response_time_ms: float
    uptime_seconds: float
    rag_quotes_count: int


# Global state
class AppState:
    def __init__(self):
        self.house_model: Optional[HouseModel] = None
        self.style_filter: Optional[StyleFilter] = None
        self.rag_store: Optional[RAGStore] = None
        self.voice_output: Optional[VoiceOutput] = None
        self.conversation_logger: Optional[ConversationLogger] = None
        self.config = None
        self.sessions: Dict[str, ConversationContext] = {}
        self.start_time = time.time()
        self.total_conversations = 0
        self.successful_conversations = 0
        self.failed_conversations = 0
        self.total_response_time = 0.0


app = FastAPI(
    title="HouseGPT Web Interface",
    description="Web API and interface for Dr. House personality AI",
    version="1.0.0"
)

state = AppState()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Static files and templates
app.mount("/static", StaticFiles(directory="web/static"), name="static")
templates = Jinja2Templates(directory="web/templates")


@app.on_event("startup")
async def startup_event():
    """Initialize HouseGPT services on startup."""
    logger.info("🏥 Starting HouseGPT Web Interface...")
    
    try:
        # Load configuration
        state.config = get_config()
        logger.info("✅ Configuration loaded")
        
        # Initialize services
        logger.info("📚 Loading House personality model...")
        state.house_model = HouseModel(state.config)
        
        logger.info("🎭 Loading style filter...")
        state.style_filter = StyleFilter(state.config)
        
        logger.info("🗄️ Loading RAG store...")
        state.rag_store = RAGStore(state.config)
        
        logger.info("🗣️ Loading voice output...")
        state.voice_output = VoiceOutput(state.config)
        
        logger.info("📝 Loading conversation logger...")
        state.conversation_logger = ConversationLogger(state.config)
        
        logger.info("✅ HouseGPT Web Interface ready!")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize services: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("🛑 Shutting down HouseGPT Web Interface...")


# Web interface routes
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main chat interface."""
    return templates.TemplateResponse("chat.html", {"request": request})


@app.get("/health")
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    uptime = time.time() - state.start_time
    
    services = {
        "house_model": state.house_model is not None,
        "style_filter": state.style_filter is not None,
        "rag_store": state.rag_store is not None,
        "voice_output": state.voice_output is not None,
        "conversation_logger": state.conversation_logger is not None,
    }
    
    all_healthy = all(services.values())
    
    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        services=services,
        uptime_seconds=uptime,
        version="1.0.0"
    )


@app.get("/stats")
async def get_stats() -> StatsResponse:
    """Get system statistics."""
    uptime = time.time() - state.start_time
    avg_response_time = (state.total_response_time / state.total_conversations 
                        if state.total_conversations > 0 else 0.0)
    
    rag_quotes_count = 0
    if state.rag_store:
        try:
            rag_stats = state.rag_store.get_stats()
            rag_quotes_count = rag_stats.get('total_quotes', 0)
        except:
            pass
    
    return StatsResponse(
        total_conversations=state.total_conversations,
        successful_conversations=state.successful_conversations,
        failed_conversations=state.failed_conversations,
        average_response_time_ms=avg_response_time,
        uptime_seconds=uptime,
        rag_quotes_count=rag_quotes_count
    )


@app.post("/chat")
async def chat(request: ChatRequest, background_tasks: BackgroundTasks) -> ChatResponse:
    """Main chat endpoint."""
    start_time = time.time()
    state.total_conversations += 1
    
    try:
        # Get or create session
        session_id = request.session_id or f"web_session_{int(time.time())}"
        
        if session_id not in state.sessions:
            state.sessions[session_id] = ConversationContext(session_id=session_id)
        
        context = state.sessions[session_id]
        
        # Create user input
        user_input = UserInput(
            text=request.message,
            session_id=session_id,
            timestamp=time.time()
        )
        
        # Add to context
        context.add_user_input(user_input)
        
        # Get RAG quotes
        rag_quotes = []
        rag_quotes_count = 0
        if state.rag_store:
            try:
                rag_quotes = state.rag_store.search_quotes(request.message, top_k=3)
                rag_quotes_count = len(rag_quotes)
                context.add_rag_context(rag_quotes)
            except Exception as e:
                logger.warning(f"RAG search failed: {e}")
        
        # Generate response
        if not state.house_model:
            raise HTTPException(status_code=503, detail="House model not available")
        
        house_response = state.house_model.generate_response(
            user_input.text, 
            context=context
        )
        
        # Apply style filtering
        if state.style_filter:
            try:
                house_response = state.style_filter.apply_style(house_response)
            except Exception as e:
                logger.warning(f"Style filtering failed: {e}")
        
        # Add to context
        context.add_house_response(house_response)
        
        # Calculate response time
        processing_time = (time.time() - start_time) * 1000
        state.total_response_time += processing_time
        state.successful_conversations += 1
        
        # Log conversation (in background)
        if state.conversation_logger:
            background_tasks.add_task(
                log_conversation_async,
                user_input,
                house_response,
                context
            )
        
        # Prepare response
        response_data = ChatResponse(
            response=house_response.text,
            session_id=session_id,
            processing_time_ms=processing_time,
            style_score=house_response.style_score.overall_score if house_response.style_score else 0.0,
            confidence=house_response.confidence,
            rag_quotes_used=rag_quotes_count,
            voice_available=state.voice_output is not None and request.voice_enabled
        )
        
        return response_data
        
    except Exception as e:
        state.failed_conversations += 1
        logger.error(f"Chat request failed: {e}")
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")


@app.post("/synthesize")
async def synthesize_voice(request: ChatRequest):
    """Synthesize voice for given text."""
    if not state.voice_output:
        raise HTTPException(status_code=503, detail="Voice synthesis not available")
    
    if not request.message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    try:
        # Create a mock HouseResponse for synthesis
        from src.models.house_response import HouseResponse
        from src.models.style_score import StyleScore
        
        mock_response = HouseResponse(
            text=request.message,
            confidence=1.0,
            processing_time=0.0,
            style_score=StyleScore()
        )
        
        # Synthesize audio
        audio_data = state.voice_output.synthesize_speech(mock_response)
        
        return StreamingResponse(
            io.BytesIO(audio_data),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=house_voice.wav"}
        )
        
    except Exception as e:
        logger.error(f"Voice synthesis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Voice synthesis failed: {str(e)}")


@app.get("/rag/search")
async def search_rag(query: str, limit: int = 5):
    """Search RAG quotes."""
    if not state.rag_store:
        raise HTTPException(status_code=503, detail="RAG store not available")
    
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        quotes = state.rag_store.search_quotes(query, top_k=limit)
        return {"quotes": quotes, "count": len(quotes)}
        
    except Exception as e:
        logger.error(f"RAG search failed: {e}")
        raise HTTPException(status_code=500, detail=f"RAG search failed: {str(e)}")


@app.get("/sessions/{session_id}/history")
async def get_conversation_history(session_id: str):
    """Get conversation history for a session."""
    if session_id not in state.sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    context = state.sessions[session_id]
    
    history = []
    for entry in context.conversation_history:
        history.append({
            "type": "user" if hasattr(entry, 'text') and not hasattr(entry, 'style_score') else "house",
            "message": entry.text,
            "timestamp": getattr(entry, 'timestamp', time.time())
        })
    
    return {"session_id": session_id, "history": history}


@app.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    """Clear a conversation session."""
    if session_id in state.sessions:
        del state.sessions[session_id]
        return {"message": f"Session {session_id} cleared"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")


async def log_conversation_async(user_input, house_response, context):
    """Async wrapper for conversation logging."""
    try:
        from src.models.conversation_entry import ConversationEntry
        
        entry = ConversationEntry(
            conversation_id=f"web_{int(time.time())}",
            user_input=user_input,
            house_response=house_response,
            context=context,
            timestamp=time.time()
        )
        
        state.conversation_logger.log_conversation_entry(entry)
        
    except Exception as e:
        logger.warning(f"Conversation logging failed: {e}")


def run_server(host: str = "127.0.0.1", port: int = 8000, reload: bool = False):
    """Run the HouseGPT web server."""
    print(f"🏥 Starting HouseGPT Web Interface on http://{host}:{port}")
    print("   Access the chat interface at: http://127.0.0.1:8000")
    print("   API documentation at: http://127.0.0.1:8000/docs")
    print("   Health check at: http://127.0.0.1:8000/health")
    
    uvicorn.run(
        "src.web.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="HouseGPT Web Interface")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    run_server(args.host, args.port, args.reload)
