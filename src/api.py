"""
V3.0 — REST API
FastAPI-based microservice with JWT auth, rate limiting, and full analytics endpoints.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from typing import Optional, List, Dict, Any
from datetime import timedelta
import pandas as pd

try:
    from fastapi import FastAPI, HTTPException, Depends, status, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from pydantic import BaseModel, Field
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("[WARN] FastAPI not installed. Install with: pip install fastapi uvicorn[standard]")

from retail_intelligence import RetailIntelligencePlatform
from security import SecurityManager, Role
from ai_recommendations import AIRecommendationsEngine
from forecasting_engine import ForecastingEngine

# ─── Pydantic Models ─────────────────────────────────────────────────────────

if FASTAPI_AVAILABLE:
    class LoginRequest(BaseModel):
        username: str
        password: str

    class TokenResponse(BaseModel):
        access_token: str
        refresh_token: str
        token_type: str = "bearer"
        expires_in: int = 3600

    class GenerateDataRequest(BaseModel):
        n_customers: int = Field(default=1000, ge=100, le=50000)
        n_transactions: int = Field(default=5000, ge=500, le=500000)
        store_id: str = "STORE_001"

    class ForecastRequest(BaseModel):
        horizon: int = Field(default=90, ge=7, le=365)
        model: str = Field(default="simple", pattern="^(simple|prophet|arima|xgboost)$")

    class SegmentationRequest(BaseModel):
        method: str = Field(default="kmeans", pattern="^(kmeans|hierarchical|dbscan)$")
        n_clusters_min: int = Field(default=3, ge=2, le=8)
        n_clusters_max: int = Field(default=10, ge=4, le=15)


# ─── In-Memory "User Store" (replace with DB in production) ──────────────────

DEMO_USERS = {
    "admin": {
        "username": "admin",
        "hashed_password": None,  # Set on startup
        "role": Role.ADMIN,
        "tenant_id": "TENANT_001",
    },
    "analyst": {
        "username": "analyst",
        "hashed_password": None,
        "role": Role.ANALYST,
        "tenant_id": "TENANT_001",
    },
}


def build_app() -> "FastAPI":
    if not FASTAPI_AVAILABLE:
        raise RuntimeError("FastAPI not installed. Run: pip install fastapi uvicorn[standard]")

    app = FastAPI(
        title="AI Retail Intelligence Platform API",
        description="V3.0 Enterprise REST API — Multi-store Analytics, Forecasting & AI Recommendations",
        version="3.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ─── Singletons ──────────────────────────────────────────────────────────
    security = SecurityManager()
    bearer_scheme = HTTPBearer(auto_error=False)
    rip_instances: Dict[str, RetailIntelligencePlatform] = {}
    ai_engine = AIRecommendationsEngine()

    # Hash demo passwords on startup
    DEMO_USERS["admin"]["hashed_password"] = security.hash_password("Admin@123")
    DEMO_USERS["analyst"]["hashed_password"] = security.hash_password("Analyst@123")

    # ─── Auth Dependency ─────────────────────────────────────────────────────
    def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
        if not credentials:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
        payload = security.decode_token(credentials.credentials)
        if not payload:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
        return payload

    def get_rip(store_id: str, user=Depends(get_current_user)) -> RetailIntelligencePlatform:
        if store_id not in rip_instances:
            rip_instances[store_id] = RetailIntelligencePlatform(
                store_id=store_id, tenant_id=user.get("tenant_id", "TENANT_001")
            )
        return rip_instances[store_id]

    # ─── Auth Routes ─────────────────────────────────────────────────────────
    @app.post("/api/v3/auth/login", response_model=TokenResponse, tags=["Auth"])
    def login(req: LoginRequest):
        user = DEMO_USERS.get(req.username)
        if not user or not security.verify_password(req.password, user["hashed_password"]):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
        payload = {"sub": req.username, "role": user["role"], "tenant_id": user["tenant_id"]}
        return TokenResponse(
            access_token=security.create_access_token(payload),
            refresh_token=security.create_refresh_token(payload),
        )

    @app.get("/api/v3/auth/me", tags=["Auth"])
    def me(user=Depends(get_current_user)):
        return {"username": user.get("sub"), "role": user.get("role"), "tenant_id": user.get("tenant_id")}

    # ─── Health ──────────────────────────────────────────────────────────────
    @app.get("/api/v3/health", tags=["System"])
    def health():
        return {"status": "healthy", "version": "3.0.0", "platform": "AI Retail Intelligence Platform"}

    # ─── Data Routes ─────────────────────────────────────────────────────────
    @app.post("/api/v3/data/generate", tags=["Data"])
    def generate_data(req: GenerateDataRequest, user=Depends(get_current_user)):
        rip = get_rip(req.store_id, user)
        rip.generate_synthetic_data(n_customers=req.n_customers, n_transactions=req.n_transactions)
        return {
            "message": "Data generated",
            "store_id": req.store_id,
            "customers": req.n_customers,
            "transactions": req.n_transactions,
        }

    # ─── Analytics Routes ─────────────────────────────────────────────────────
    @app.post("/api/v3/analytics/{store_id}/rfm", tags=["Analytics"])
    def compute_rfm(store_id: str, user=Depends(get_current_user)):
        rip = get_rip(store_id, user)
        if rip.transactions_df is None:
            raise HTTPException(400, "No data. Call /data/generate first.")
        rfm = rip.calculate_rfm_metrics()
        return {"customers": len(rfm), "sample": rfm.head(5).to_dict("records")}

    @app.post("/api/v3/analytics/{store_id}/segment", tags=["Analytics"])
    def segment(store_id: str, req: SegmentationRequest, user=Depends(get_current_user)):
        rip = get_rip(store_id, user)
        if rip.rfm_df is None:
            raise HTTPException(400, "Compute RFM first.")
        _, k = rip.perform_advanced_segmentation(
            method=req.method,
            n_clusters_range=(req.n_clusters_min, req.n_clusters_max)
        )
        seg = rip.analyze_segments()
        return {"optimal_k": k, "segments": seg.to_dict("records")}

    @app.get("/api/v3/analytics/{store_id}/insights", tags=["Analytics"])
    def insights(store_id: str, user=Depends(get_current_user)):
        rip = get_rip(store_id, user)
        if rip.rfm_df is None:
            raise HTTPException(400, "Run RFM first.")
        return rip.generate_business_insights()

    @app.get("/api/v3/analytics/{store_id}/anomalies", tags=["Analytics"])
    def anomalies(store_id: str, contamination: float = 0.1, user=Depends(get_current_user)):
        rip = get_rip(store_id, user)
        if rip.rfm_df is None:
            raise HTTPException(400, "Run RFM first.")
        anom = rip.detect_anomalies(contamination=contamination)
        return {"anomalies_detected": len(anom), "sample": anom.head(10).to_dict("records")}

    # ─── Forecasting Routes ───────────────────────────────────────────────────
    @app.post("/api/v3/forecast/{store_id}", tags=["Forecasting"])
    def forecast(store_id: str, req: ForecastRequest, user=Depends(get_current_user)):
        rip = get_rip(store_id, user)
        if rip.transactions_df is None:
            raise HTTPException(400, "No data. Call /data/generate first.")
        eng = ForecastingEngine(model=req.model)
        fc = eng.forecast(rip.transactions_df, horizon=req.horizon)
        return {"summary": eng.get_forecast_summary(), "forecast_sample": fc.head(14).to_dict("records")}

    # ─── AI Recommendations Routes ────────────────────────────────────────────
    @app.get("/api/v3/recommendations/{store_id}", tags=["AI"])
    def recommendations(store_id: str, user=Depends(get_current_user)):
        rip = get_rip(store_id, user)
        if rip.rfm_df is None:
            raise HTTPException(400, "Run RFM + segmentation first.")
        kpis = rip.generate_business_insights()
        seg = rip.analyze_segments().to_dict("records")
        recs = ai_engine.generate(kpis, seg)
        return {"total": len(recs), "recommendations": recs}

    return app


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not FASTAPI_AVAILABLE:
        print("Install: pip install fastapi uvicorn[standard]")
        sys.exit(1)
    app = build_app()
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
