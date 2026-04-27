"""Tests for Forecasting Engine — V3.0"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import pytest
import pandas as pd
from retail_intelligence import RetailIntelligencePlatform
from forecasting_engine import ForecastingEngine


@pytest.fixture
def transactions():
    r = RetailIntelligencePlatform()
    r.generate_synthetic_data(n_customers=200, n_transactions=800)
    return r.transactions_df

def test_simple_forecast_returns_df(transactions):
    eng = ForecastingEngine(model='simple')
    fc = eng.forecast(transactions, horizon=30)
    assert isinstance(fc, pd.DataFrame)
    assert len(fc) == 30
    assert 'yhat' in fc.columns

def test_forecast_positive_values(transactions):
    eng = ForecastingEngine(model='simple')
    fc = eng.forecast(transactions, horizon=30)
    assert (fc['yhat'] >= 0).all()

def test_forecast_summary_keys(transactions):
    eng = ForecastingEngine(model='simple')
    eng.forecast(transactions, horizon=30)
    summary = eng.get_forecast_summary()
    for key in ['horizon_days', 'total_projected_revenue', 'avg_daily_revenue', 'peak_day']:
        assert key in summary

def test_forecast_horizon_30(transactions):
    eng = ForecastingEngine(model='simple')
    fc = eng.forecast(transactions, horizon=30)
    assert len(fc) == 30

def test_forecast_horizon_90(transactions):
    eng = ForecastingEngine(model='simple')
    fc = eng.forecast(transactions, horizon=90)
    assert len(fc) == 90
