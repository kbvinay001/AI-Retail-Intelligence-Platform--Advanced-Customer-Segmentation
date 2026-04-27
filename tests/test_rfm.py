"""Tests for RFM calculation — V3.0"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import pytest
import pandas as pd
from retail_intelligence import RetailIntelligencePlatform


@pytest.fixture
def rip():
    r = RetailIntelligencePlatform(store_id="TEST_STORE")
    r.generate_synthetic_data(n_customers=100, n_transactions=300)
    return r

def test_generate_data(rip):
    assert rip.customers_df is not None
    assert len(rip.customers_df) == 100
    assert rip.transactions_df is not None
    assert len(rip.transactions_df) == 300

def test_rfm_columns(rip):
    rfm = rip.calculate_rfm_metrics()
    for col in ['customer_id', 'recency', 'frequency', 'monetary', 'estimated_clv', 'loyalty_score']:
        assert col in rfm.columns, f"Missing column: {col}"

def test_rfm_values_valid(rip):
    rfm = rip.calculate_rfm_metrics()
    assert (rfm['recency'] >= 0).all()
    assert (rfm['frequency'] > 0).all()
    assert (rfm['monetary'] > 0).all()
    assert (rfm['loyalty_score'] >= 0).all()

def test_rfm_customer_count(rip):
    rfm = rip.calculate_rfm_metrics()
    assert len(rfm) <= 100  # ≤ n_customers (some may have no transactions)

def test_clv_positive(rip):
    rfm = rip.calculate_rfm_metrics()
    assert (rfm['estimated_clv'] >= 0).all()
