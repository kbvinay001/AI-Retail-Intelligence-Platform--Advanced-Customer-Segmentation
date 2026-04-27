"""
V3.0 — Multi-Store Analytics Engine
Cross-location KPI comparison, store benchmarking, and consolidated reporting.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from retail_intelligence import RetailIntelligencePlatform


@dataclass
class StoreConfig:
    store_id: str
    name: str
    city: str
    region: str
    tier: str = "standard"           # 'flagship' | 'standard' | 'express'
    opened_date: str = "2020-01-01"
    currency: str = "INR"
    timezone: str = "Asia/Kolkata"
    metadata: dict = field(default_factory=dict)


class MultiStoreAnalytics:
    """
    Manages multiple store instances under a single tenant.
    Provides consolidated KPIs, cross-store benchmarking, and
    region-level roll-up reports.
    """

    def __init__(self, tenant_id: str = "TENANT_001"):
        self.tenant_id = tenant_id
        self.stores: Dict[str, RetailIntelligencePlatform] = {}
        self.store_configs: Dict[str, StoreConfig] = {}
        self._consolidated_rfm: Optional[pd.DataFrame] = None

    # ─── Store Management ────────────────────────────────────────────────────

    def register_store(self, config: StoreConfig) -> RetailIntelligencePlatform:
        """Register a new store and return its platform instance."""
        rip = RetailIntelligencePlatform(store_id=config.store_id, tenant_id=self.tenant_id)
        self.stores[config.store_id] = rip
        self.store_configs[config.store_id] = config
        print(f"[STORE] Registered: {config.name} ({config.store_id}) -- {config.city}")
        return rip

    def get_store(self, store_id: str) -> Optional[RetailIntelligencePlatform]:
        return self.stores.get(store_id)

    def list_stores(self) -> List[dict]:
        result = []
        for sid, cfg in self.store_configs.items():
            result.append({
                'store_id': sid,
                'name': cfg.name,
                'city': cfg.city,
                'region': cfg.region,
                'tier': cfg.tier,
            })
        return result

    # ─── Populate Demo Data ──────────────────────────────────────────────────

    def populate_all_stores(self, customers_per_store=500, txn_per_store=2500):
        """Generate synthetic data for all registered stores."""
        for store_id, rip in self.stores.items():
            rip.generate_synthetic_data(
                n_customers=customers_per_store,
                n_transactions=txn_per_store
            )
            rip.calculate_rfm_metrics()
            rip.perform_advanced_segmentation()
        print(f"[OK] Data populated across {len(self.stores)} stores.")

    # ─── Consolidated KPIs ───────────────────────────────────────────────────

    def get_consolidated_kpis(self) -> pd.DataFrame:
        """Return a DataFrame of KPIs per store."""
        rows = []
        for store_id, rip in self.stores.items():
            if rip.transactions_df is None:
                continue
            cfg = self.store_configs[store_id]
            txn = rip.transactions_df
            rfm = rip.rfm_df if rip.rfm_df is not None else pd.DataFrame()

            rows.append({
                'store_id': store_id,
                'store_name': cfg.name,
                'city': cfg.city,
                'region': cfg.region,
                'tier': cfg.tier,
                'total_customers': len(rfm) if not rfm.empty else 0,
                'total_transactions': len(txn),
                'total_revenue': round(txn['amount'].sum(), 2),
                'avg_order_value': round(txn['amount'].mean(), 2),
                'avg_clv': round(rfm['estimated_clv'].mean(), 2) if not rfm.empty else 0,
                'total_clv': round(rfm['estimated_clv'].sum(), 2) if not rfm.empty else 0,
            })
        return pd.DataFrame(rows)

    # ─── Benchmarking ────────────────────────────────────────────────────────

    def benchmark_stores(self) -> pd.DataFrame:
        """Rank stores by revenue, AOV, and CLV with percentile scores."""
        kpis = self.get_consolidated_kpis()
        if kpis.empty:
            return kpis

        for col in ['total_revenue', 'avg_order_value', 'avg_clv']:
            kpis[f'{col}_rank'] = kpis[col].rank(ascending=False).astype(int)
            kpis[f'{col}_pct'] = kpis[col].rank(pct=True).round(2) * 100

        kpis['composite_score'] = (
            kpis['total_revenue'] / kpis['total_revenue'].max() * 0.5 +
            kpis['avg_clv'] / kpis['avg_clv'].max() * 0.3 +
            kpis['avg_order_value'] / kpis['avg_order_value'].max() * 0.2
        ) * 100

        return kpis.sort_values('composite_score', ascending=False).reset_index(drop=True)

    # ─── Regional Roll-Up ────────────────────────────────────────────────────

    def regional_rollup(self) -> pd.DataFrame:
        """Aggregate KPIs by region."""
        kpis = self.get_consolidated_kpis()
        if kpis.empty:
            return kpis
        return kpis.groupby('region').agg(
            stores=('store_id', 'count'),
            total_revenue=('total_revenue', 'sum'),
            total_customers=('total_customers', 'sum'),
            avg_aov=('avg_order_value', 'mean'),
            avg_clv=('avg_clv', 'mean'),
        ).reset_index().sort_values('total_revenue', ascending=False)

    # ─── Consolidated RFM ────────────────────────────────────────────────────

    def get_consolidated_rfm(self) -> pd.DataFrame:
        """Merge RFM data from all stores into one DataFrame."""
        frames = []
        for store_id, rip in self.stores.items():
            if rip.rfm_df is not None:
                df = rip.rfm_df.copy()
                df['store_id'] = store_id
                frames.append(df)
        if frames:
            self._consolidated_rfm = pd.concat(frames, ignore_index=True)
        return self._consolidated_rfm

    # ─── Top Customers Across All Stores ─────────────────────────────────────

    def top_customers_network(self, n: int = 20) -> pd.DataFrame:
        """Return top N customers by CLV across all stores."""
        rfm = self.get_consolidated_rfm()
        if rfm is None or rfm.empty:
            return pd.DataFrame()
        return rfm.nlargest(n, 'estimated_clv')[
            ['customer_id', 'store_id', 'estimated_clv', 'frequency', 'recency', 'loyalty_score']
        ].reset_index(drop=True)

    # ─── Print Summary ───────────────────────────────────────────────────────

    def print_network_summary(self):
        kpis = self.get_consolidated_kpis()
        bench = self.benchmark_stores()
        regional = self.regional_rollup()

        print("\n" + "=" * 60)
        print(f"  MULTI-STORE NETWORK SUMMARY | Tenant: {self.tenant_id}")
        print("=" * 60)
        print(f"\n  STORES ({len(self.stores)} registered):")
        for _, row in kpis.iterrows():
            print(f"   [{row['store_id']}] {row['store_name']} -- "
                  f"Revenue: ${row['total_revenue']:,.0f} | "
                  f"Customers: {row['total_customers']:,}")

        print("\n  BENCHMARKING (Top Performers):")
        for _, row in bench.head(3).iterrows():
            print(f"   #{int(row['total_revenue_rank'])} {row['store_name']}: "
                  f"Score {row['composite_score']:.1f}/100")

        print("\n  REGIONAL ROLL-UP:")
        for _, row in regional.iterrows():
            print(f"   {row['region']}: ${row['total_revenue']:,.0f} across {row['stores']} stores")
        print("=" * 60)
