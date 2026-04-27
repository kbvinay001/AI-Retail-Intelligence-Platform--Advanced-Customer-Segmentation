"""Tests for segmentation & anomaly detection — V3.0"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import pytest
from retail_intelligence import RetailIntelligencePlatform


@pytest.fixture
def rip_with_rfm():
    r = RetailIntelligencePlatform(store_id="TEST_STORE")
    r.generate_synthetic_data(n_customers=200, n_transactions=600)
    r.calculate_rfm_metrics()
    return r

def test_kmeans_segmentation(rip_with_rfm):
    _, k = rip_with_rfm.perform_advanced_segmentation(method='kmeans', n_clusters_range=(3, 6))
    assert 3 <= k <= 6
    assert 'segment' in rip_with_rfm.rfm_df.columns

def test_segment_names_assigned(rip_with_rfm):
    rip_with_rfm.perform_advanced_segmentation()
    assert 'segment_name' in rip_with_rfm.rfm_df.columns
    assert rip_with_rfm.rfm_df['segment_name'].notna().all()

def test_anomaly_detection(rip_with_rfm):
    anom = rip_with_rfm.detect_anomalies(contamination=0.10)
    total = len(rip_with_rfm.rfm_df)
    assert len(anom) <= total * 0.15  # Should not exceed 15%

def test_segment_analysis(rip_with_rfm):
    rip_with_rfm.perform_advanced_segmentation()
    seg = rip_with_rfm.analyze_segments()
    assert 'count' in seg.columns
    assert 'avg_monetary' in seg.columns
    assert seg['count'].sum() == len(rip_with_rfm.rfm_df)
