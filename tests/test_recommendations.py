"""Tests for AI Recommendations Engine — V3.0"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import pytest
from ai_recommendations import AIRecommendationsEngine


@pytest.fixture
def engine():
    return AIRecommendationsEngine(api_key=None)  # Rule-based mode

@pytest.fixture
def sample_kpis():
    return {
        'total_customers': 1000,
        'total_revenue': 250000.0,
        'avg_order_value': 45.0,  # Low AOV triggers rule
        'total_clv': 800000.0,
        'avg_clv': 800.0,
        'at_risk_customers': 300,  # 30% — triggers churn rule
        'high_value_customers': 200,  # 20% — triggers VIP rule
    }

def test_rule_engine_returns_list(engine, sample_kpis):
    recs = engine.generate(sample_kpis, [])
    assert isinstance(recs, list)
    assert len(recs) > 0

def test_recommendations_have_required_fields(engine, sample_kpis):
    recs = engine.generate(sample_kpis, [])
    for r in recs:
        assert 'title' in r
        assert 'recommendation' in r
        assert 'priority' in r
        assert 'actions' in r

def test_churn_rule_triggered(engine, sample_kpis):
    recs = engine.generate(sample_kpis, [])
    ids = [r['id'] for r in recs]
    assert 'high_churn_risk' in ids

def test_low_aov_rule_triggered(engine, sample_kpis):
    recs = engine.generate(sample_kpis, [])
    ids = [r['id'] for r in recs]
    assert 'low_aov' in ids

def test_priority_valid_values(engine, sample_kpis):
    recs = engine.generate(sample_kpis, [])
    valid = {'CRITICAL', 'HIGH', 'MEDIUM', 'LOW'}
    for r in recs:
        assert r['priority'] in valid
