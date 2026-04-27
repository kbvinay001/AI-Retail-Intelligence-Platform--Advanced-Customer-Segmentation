"""Tests for Security module — V3.0"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import pytest
from security import SecurityManager, Role, ROLE_PERMISSIONS


@pytest.fixture
def sec():
    return SecurityManager()

def test_password_hash_and_verify(sec):
    pwd = "Test@1234"
    hashed = sec.hash_password(pwd)
    assert hashed != pwd
    assert sec.verify_password(pwd, hashed)
    assert not sec.verify_password("Wrong@999", hashed)

def test_password_policy_valid(sec):
    result = sec.validate_password_policy("Admin@123")
    assert all(result.values())

def test_password_policy_too_short(sec):
    result = sec.validate_password_policy("Ab1!")
    assert not result["min_length"]

def test_jwt_round_trip(sec):
    token = sec.create_access_token({"sub": "admin", "role": "admin"})
    payload = sec.decode_token(token)
    assert payload is not None
    assert payload["sub"] == "admin"

def test_rbac_admin_has_all(sec):
    for perm in ["read", "write", "delete", "admin"]:
        assert sec.has_permission(Role.ADMIN, perm)

def test_rbac_viewer_read_only(sec):
    assert sec.has_permission(Role.VIEWER, "read")
    assert not sec.has_permission(Role.VIEWER, "write")
    assert not sec.has_permission(Role.VIEWER, "delete")

def test_field_encryption(sec):
    secret = "customer@email.com"
    enc = sec.encrypt_field(secret)
    dec = sec.decrypt_field(enc)
    assert dec == secret

def test_api_key_generation(sec):
    key = SecurityManager.generate_api_key()
    assert key.startswith("ari_")
    assert len(key) > 10
