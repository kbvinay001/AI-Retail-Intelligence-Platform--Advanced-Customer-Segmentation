"""Tests for API endpoints — V3.0"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import pytest

try:
    from fastapi.testclient import TestClient
    from api import build_app
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

pytestmark = pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")


@pytest.fixture
def client():
    app = build_app()
    return TestClient(app)

def get_token(client):
    resp = client.post("/api/v3/auth/login", json={"username": "admin", "password": "Admin@123"})
    return resp.json()["access_token"]

def test_health(client):
    resp = client.get("/api/v3/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"

def test_login_success(client):
    resp = client.post("/api/v3/auth/login", json={"username": "admin", "password": "Admin@123"})
    assert resp.status_code == 200
    assert "access_token" in resp.json()

def test_login_wrong_password(client):
    resp = client.post("/api/v3/auth/login", json={"username": "admin", "password": "wrong"})
    assert resp.status_code == 401

def test_me_endpoint(client):
    token = get_token(client)
    resp = client.get("/api/v3/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    assert resp.json()["username"] == "admin"

def test_generate_data(client):
    token = get_token(client)
    resp = client.post("/api/v3/data/generate",
                       json={"n_customers": 200, "n_transactions": 500, "store_id": "TEST"},
                       headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200

def test_rfm_endpoint(client):
    token = get_token(client)
    client.post("/api/v3/data/generate",
                json={"n_customers": 200, "n_transactions": 500, "store_id": "S1"},
                headers={"Authorization": f"Bearer {token}"})
    resp = client.post("/api/v3/analytics/S1/rfm",
                       headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    assert "customers" in resp.json()
