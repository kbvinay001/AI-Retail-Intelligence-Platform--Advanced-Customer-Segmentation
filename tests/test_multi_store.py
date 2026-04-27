"""Tests for Multi-Store Analytics — V3.0"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import pytest
from multi_store import MultiStoreAnalytics, StoreConfig


@pytest.fixture
def network():
    ms = MultiStoreAnalytics(tenant_id="TEST_TENANT")
    for i, (city, region) in enumerate([("Mumbai","West"), ("Delhi","North"), ("Chennai","South")], 1):
        cfg = StoreConfig(store_id=f"S{i:03}", name=f"Store {i}", city=city, region=region)
        ms.register_store(cfg)
    ms.populate_all_stores(customers_per_store=100, txn_per_store=300)
    return ms

def test_store_count(network):
    assert len(network.stores) == 3

def test_list_stores(network):
    lst = network.list_stores()
    assert len(lst) == 3
    for s in lst:
        assert 'store_id' in s
        assert 'region' in s

def test_consolidated_kpis(network):
    kpis = network.get_consolidated_kpis()
    assert len(kpis) == 3
    assert 'total_revenue' in kpis.columns
    assert (kpis['total_revenue'] > 0).all()

def test_benchmark_stores(network):
    bench = network.benchmark_stores()
    assert 'composite_score' in bench.columns
    assert bench['composite_score'].max() <= 100

def test_regional_rollup(network):
    regional = network.regional_rollup()
    assert len(regional) == 3  # 3 distinct regions

def test_top_customers_network(network):
    top = network.top_customers_network(n=10)
    assert len(top) == 10
