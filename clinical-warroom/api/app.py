"""
Clinical War Room - FastAPI Application

Main API application instance and configuration.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from core.config import settings
from core.logging import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("Clinical War Room starting up...")
    settings.validate()
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Log level: {settings.log_level}")
    
    yield
    
    # Shutdown
    logger.info("Clinical War Room shutting down...")


# Create FastAPI application
app = FastAPI(
    title="Clinical War Room",
    description="""
    Multi-agent clinical decision support system.
    
    This system simulates a hospital clinical board meeting where
    multiple AI specialists review patient cases, debate, and decide
    whether an action should be taken, escalated, or refused.
    
    **Key Principles:**
    - The system outputs recommendations, NOT treatments
    - Uncertainty leads to escalation, not forced decisions
    - Human clinicians can always override
    - All decisions are explainable
    """,
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "environment": settings.environment,
        "version": "0.1.0",
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Clinical War Room",
        "version": "0.1.0",
        "docs": "/docs",
        "status": "Phase 0 - Infrastructure Only",
    }


# Import and include routers
from api.routes import case, decision, human_review

app.include_router(case.router, prefix="/api/v1", tags=["Cases"])
app.include_router(decision.router, prefix="/api/v1", tags=["Decisions"])
app.include_router(human_review.router, prefix="/api/v1", tags=["Human Review"])
