"""
AI Retail Intelligence Platform V3.0
Centralized Configuration Management with Security & Multi-store Support
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ─── Base Paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
MODELS_DIR = BASE_DIR / "models"
EXPORTS_DIR = BASE_DIR / "exports"

for _d in [DATA_DIR, LOGS_DIR, MODELS_DIR, EXPORTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ─── Application Settings ────────────────────────────────────────────────────
APP_NAME = "AI Retail Intelligence Platform"
APP_VERSION = "3.0.0"
APP_DESCRIPTION = (
    "Enterprise-grade AI-powered retail analytics with multi-store support, "
    "REST API, advanced forecasting, and GPT-powered recommendations."
)
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

# ─── Security (V3.0) ─────────────────────────────────────────────────────────
SECRET_KEY = os.getenv("SECRET_KEY", "ai-retail-v3-super-secret-key-change-in-prod-2024")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "30"))
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY", None)  # Fernet key for field-level encryption

# Password Policy
MIN_PASSWORD_LENGTH = 8
REQUIRE_UPPERCASE = True
REQUIRE_DIGIT = True
REQUIRE_SPECIAL_CHAR = True

# Rate Limiting
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))

# ─── Database (V3.0) ─────────────────────────────────────────────────────────
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"sqlite:///{DATA_DIR / 'retail_intelligence.db'}"
)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
DB_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "10"))
DB_MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "20"))

# ─── API Settings (V3.0) ─────────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_PREFIX = "/api/v3"
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8080").split(",")
DOCS_URL = "/docs"
REDOC_URL = "/redoc"

# ─── OpenAI / GPT (V3.0) ─────────────────────────────────────────────────────
OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY", None)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
AI_RECOMMENDATIONS_ENABLED = OPENAI_API_KEY is not None

# ─── Multi-store (V3.0) ──────────────────────────────────────────────────────
DEFAULT_STORE_ID = "STORE_001"
MAX_STORES_PER_TENANT = int(os.getenv("MAX_STORES_PER_TENANT", "50"))
STORE_TIMEZONE = os.getenv("STORE_TIMEZONE", "UTC")

# ─── Email Notifications (V3.0) ──────────────────────────────────────────────
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
NOTIFICATION_FROM = os.getenv("NOTIFICATION_FROM", "alerts@retail-intelligence.ai")

# ─── ML / Segmentation ───────────────────────────────────────────────────────
DEFAULT_N_CLUSTERS = 6
CLUSTER_RANGE = (3, 12)
DEFAULT_CONTAMINATION = 0.10   # Anomaly detection
DEFAULT_CLV_DAYS = 365
RANDOM_STATE = 42

# ─── Forecasting (V3.0) ──────────────────────────────────────────────────────
FORECAST_HORIZON_DAYS = 90
SEASONALITY_MODES = ["additive", "multiplicative"]
FORECAST_CONFIDENCE_INTERVAL = 0.95
DEFAULT_FORECAST_MODEL = "prophet"  # 'prophet' | 'arima' | 'xgboost'

# ─── Celery (Async Tasks) ────────────────────────────────────────────────────
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", REDIS_URL)
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", REDIS_URL)

# ─── Logging ─────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = LOGS_DIR / "platform.log"
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
LOG_ROTATION = "10 MB"
LOG_RETENTION = "30 days"
