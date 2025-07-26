"""
FastAPI application entry point with hot-reload configuration
"""
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from app.api.routes import cases, documents, search, admin, websocket
from app.core.config_watcher import ConfigWatcher
from app.core.database import init_databases
from app.core.websocket_manager import WebSocketManager


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await init_databases()
    config_watcher = ConfigWatcher()
    config_watcher.start()
    
    yield
    
    # Shutdown
    config_watcher.stop()


app = FastAPI(
    title="Patexia Legal AI API",
    description="AI-powered legal document processing and search",
    version="1.0.0",
    lifespan=lifespan
)

# Include routers
app.include_router(cases.router, prefix="/api/v1/cases", tags=["cases"])
app.include_router(documents.router, prefix="/api/v1/documents", tags=["documents"])
app.include_router(search.router, prefix="/api/v1/search", tags=["search"])
app.include_router(admin.router, prefix="/api/v1/admin", tags=["admin"])
app.include_router(websocket.router, prefix="/ws", tags=["websocket"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_config="config/logging.json"
    )