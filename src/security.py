"""
V3.0 — Security Module
JWT auth, password hashing, field-level encryption, RBAC.
"""

import os
import re
import hashlib
import secrets
import base64
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from enum import Enum

try:
    from jose import JWTError, jwt
    JOSE_AVAILABLE = True
except ImportError:
    JOSE_AVAILABLE = False

try:
    from passlib.context import CryptContext
    # bcrypt on some systems has a 72-byte bug; use sha256_crypt as safe fallback
    try:
        _test_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
        _test_ctx.hash("test")
        pwd_context = _test_ctx
    except Exception:
        pwd_context = CryptContext(schemes=["sha256_crypt"], deprecated="auto")
    PASSLIB_AVAILABLE = True
except ImportError:
    PASSLIB_AVAILABLE = False

try:
    from cryptography.fernet import Fernet
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False


# ─── RBAC Roles ──────────────────────────────────────────────────────────────

class Role(str, Enum):
    ADMIN = "admin"
    ANALYST = "analyst"
    VIEWER = "viewer"
    STORE_MANAGER = "store_manager"

ROLE_PERMISSIONS: Dict[str, List[str]] = {
    Role.ADMIN:         ["read", "write", "delete", "admin", "export", "manage_users"],
    Role.ANALYST:       ["read", "write", "export"],
    Role.STORE_MANAGER: ["read", "write", "export"],
    Role.VIEWER:        ["read"],
}


class SecurityManager:
    """
    Handles JWT token issuance/validation, password operations,
    field-level encryption, and RBAC permission checks.
    """

    SECRET_KEY = os.getenv("SECRET_KEY", "ai-retail-v3-secret-change-in-production")
    ALGORITHM  = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES  = 60
    REFRESH_TOKEN_EXPIRE_DAYS    = 30

    def __init__(self, encryption_key: Optional[str] = None):
        if CRYPTOGRAPHY_AVAILABLE:
            raw_key = encryption_key or os.getenv("ENCRYPTION_KEY")
            if raw_key:
                self._fernet = Fernet(raw_key.encode() if isinstance(raw_key, str) else raw_key)
            else:
                key = Fernet.generate_key()
                self._fernet = Fernet(key)
                print(f"[WARN] No ENCRYPTION_KEY set. Generated ephemeral key (non-persistent).")
        else:
            self._fernet = None

    # ─── Password ─────────────────────────────────────────────────────────

    def hash_password(self, plain: str) -> str:
        if PASSLIB_AVAILABLE:
            return pwd_context.hash(plain)
        # Fallback: SHA-256 + salt (not production-grade)
        salt = secrets.token_hex(16)
        return f"sha256${salt}${hashlib.sha256((salt + plain).encode()).hexdigest()}"

    def verify_password(self, plain: str, hashed: str) -> bool:
        if PASSLIB_AVAILABLE:
            return pwd_context.verify(plain, hashed)
        if hashed.startswith("sha256$"):
            _, salt, digest = hashed.split("$")
            return hashlib.sha256((salt + plain).encode()).hexdigest() == digest
        return False

    def validate_password_policy(self, password: str) -> Dict[str, bool]:
        return {
            "min_length":     len(password) >= 8,
            "has_uppercase":  bool(re.search(r"[A-Z]", password)),
            "has_digit":      bool(re.search(r"\d", password)),
            "has_special":    bool(re.search(r"[!@#$%^&*(),.?\":{}|<>]", password)),
        }

    def is_password_valid(self, password: str) -> bool:
        return all(self.validate_password_policy(password).values())

    # ─── JWT ──────────────────────────────────────────────────────────────

    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        payload = data.copy()
        expire = datetime.utcnow() + (expires_delta or timedelta(minutes=self.ACCESS_TOKEN_EXPIRE_MINUTES))
        payload.update({"exp": expire, "type": "access"})
        if JOSE_AVAILABLE:
            return jwt.encode(payload, self.SECRET_KEY, algorithm=self.ALGORITHM)
        # Fallback: base64-encoded JSON (NOT secure — demo only)
        import json
        return base64.urlsafe_b64encode(json.dumps(payload, default=str).encode()).decode()

    def create_refresh_token(self, data: dict) -> str:
        payload = data.copy()
        expire = datetime.utcnow() + timedelta(days=self.REFRESH_TOKEN_EXPIRE_DAYS)
        payload.update({"exp": expire, "type": "refresh"})
        if JOSE_AVAILABLE:
            return jwt.encode(payload, self.SECRET_KEY, algorithm=self.ALGORITHM)
        import json
        return base64.urlsafe_b64encode(json.dumps(payload, default=str).encode()).decode()

    def decode_token(self, token: str) -> Optional[dict]:
        if JOSE_AVAILABLE:
            try:
                return jwt.decode(token, self.SECRET_KEY, algorithms=[self.ALGORITHM])
            except JWTError:
                return None
        try:
            import json
            return json.loads(base64.urlsafe_b64decode(token.encode()).decode())
        except Exception:
            return None

    # ─── Field-Level Encryption ───────────────────────────────────────────

    def encrypt_field(self, value: str) -> str:
        if self._fernet:
            return self._fernet.encrypt(value.encode()).decode()
        return value  # No-op if cryptography not installed

    def decrypt_field(self, value: str) -> str:
        if self._fernet:
            try:
                return self._fernet.decrypt(value.encode()).decode()
            except Exception:
                return value
        return value

    # ─── RBAC ─────────────────────────────────────────────────────────────

    def has_permission(self, role: str, permission: str) -> bool:
        return permission in ROLE_PERMISSIONS.get(role, [])

    def get_permissions(self, role: str) -> List[str]:
        return ROLE_PERMISSIONS.get(role, [])

    # ─── API Key ──────────────────────────────────────────────────────────

    @staticmethod
    def generate_api_key(prefix: str = "ari") -> str:
        return f"{prefix}_{secrets.token_urlsafe(32)}"

    @staticmethod
    def hash_api_key(api_key: str) -> str:
        return hashlib.sha256(api_key.encode()).hexdigest()
