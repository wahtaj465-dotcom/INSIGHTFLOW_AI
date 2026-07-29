from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import router as analytics_router

# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(

    title="InsightFlow AI",

    description=(
        "Generative AI powered analytics platform "
        "for automated data preprocessing, EDA, "
        "natural-language SQL generation and "
        "AI-powered insight generation."
    ),

    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(

    CORSMiddleware,

    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173"
    ],

    allow_credentials=True,

    allow_methods=[
        "*"
    ],

    allow_headers=[
        "*"
    ]
)


# ============================================================
# REGISTER API ROUTES
# ============================================================

app.include_router(
    analytics_router
)


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():
    """
    Basic endpoint used to verify that
    the InsightFlow API is running.
    """

    return {

        "message":
            "InsightFlow AI API is running",

        "version":
            "1.0.0",

        "docs":
            "/docs"
    }


# ============================================================
# APPLICATION HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():
    """
    Application-level health check.
    """

    return {
        "status": "healthy"
    }