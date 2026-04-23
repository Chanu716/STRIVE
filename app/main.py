"""FastAPI application entrypoint for the STRIVE risk API."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app.db.session import SessionLocal
from app.ml.inference import load_model
from app.routers.explain import router as explain_router
from app.routers.risk import router as risk_router
from app.routers.route import router as route_router


app = FastAPI(
    title="STRIVE Risk API",
    version="0.1.0",
    description=(
        "Backend API for STRIVE road-segment risk scoring. "
        "This isolated implementation covers the M3 risk endpoints and uses "
        "local graph/data fallbacks until the upstream model and DB tasks land."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(risk_router)
app.include_router(explain_router)
app.include_router(route_router)


@app.get(
    "/health",
    summary="Check API Health",
    description="Verify that the API can load the model layer and connect to the configured database.",
)
def health() -> dict[str, bool | str]:
    model_loaded = False
    db_connected = False

    try:
        load_model()
        model_loaded = True
    except Exception:
        model_loaded = False

    try:
        with SessionLocal() as session:
            session.execute(text("SELECT 1"))
        db_connected = True
    except Exception:
        db_connected = False

    if not model_loaded or not db_connected:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "unavailable",
                "model_loaded": model_loaded,
                "db_connected": db_connected,
            },
        )

    return {"status": "ok", "model_loaded": model_loaded, "db_connected": db_connected}
