"""
AI Retail Intelligence Platform - V3.0
Main Platform Engine
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class RetailIntelligencePlatform:
    """
    AI Retail Intelligence Platform V3.0
    Supports multi-store, enterprise analytics, forecasting, REST API and security.
    """

    def __init__(self, store_id: str = "STORE_001", tenant_id: str = "TENANT_001"):
        self.store_id = store_id
        self.tenant_id = tenant_id
        self.customers_df = None
        self.transactions_df = None
        self.rfm_df = None
        self.segment_labels = None
        self.optimal_k = None
        self.anomalies_df = None
        self.scaler = RobustScaler()
        random.seed(42)
        np.random.seed(42)
        print(f"[OK] AI Retail Intelligence Platform V3.0 initialized")
        print(f"     Store: {store_id} | Tenant: {tenant_id}")

    # ─── Synthetic Data Generation ───────────────────────────────────────────

    def generate_synthetic_data(self, n_customers=1000, n_transactions=5000):
        """Generate realistic synthetic retail data."""
        categories = ['Electronics', 'Clothing', 'Home & Garden', 'Groceries', 'Sports', 'Beauty']
        cities = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Hyderabad', 'Pune']
        channels = ['Online', 'In-Store', 'Mobile App']

        customers = []
        for cid in range(1, n_customers + 1):
            customers.append({
                'customer_id': cid,
                'age_group': random.choice(['18-25', '26-35', '36-45', '46-55', '55+']),
                'gender': random.choice(['Male', 'Female']),
                'city': random.choice(cities),
                'acquisition_date': (datetime.now() - timedelta(days=random.randint(30, 1095))).strftime('%Y-%m-%d'),
                'store_id': self.store_id,
                'tenant_id': self.tenant_id,
            })
        self.customers_df = pd.DataFrame(customers)

        transactions = []
        for _ in range(n_transactions):
            cid = random.randint(1, n_customers)
            cat = random.choice(categories)
            amount = round(random.lognormvariate(4.0, 0.8), 2)
            transactions.append({
                'transaction_id': _ + 1,
                'customer_id': cid,
                'transaction_date': (datetime.now() - timedelta(days=random.randint(1, 730))).strftime('%Y-%m-%d'),
                'amount': amount,
                'quantity': random.randint(1, 5),
                'category': cat,
                'channel': random.choice(channels),
                'store_id': self.store_id,
            })
        self.transactions_df = pd.DataFrame(transactions)
        self.transactions_df['transaction_date'] = pd.to_datetime(self.transactions_df['transaction_date'])

        print(f"[DATA] Generated {n_customers} customers, {n_transactions} transactions")
        return self.customers_df, self.transactions_df

    # ─── RFM Metrics ─────────────────────────────────────────────────────────

    def calculate_rfm_metrics(self):
        """Calculate RFM + CLV metrics."""
        if self.transactions_df is None:
            raise ValueError("No transaction data. Run generate_synthetic_data() first.")

        ref_date = self.transactions_df['transaction_date'].max() + timedelta(days=1)
        rfm = self.transactions_df.groupby('customer_id').agg(
            recency=('transaction_date', lambda x: (ref_date - x.max()).days),
            frequency=('transaction_id', 'count'),
            monetary=('amount', 'sum'),
        ).reset_index()

        rfm['avg_order_value'] = rfm['monetary'] / rfm['frequency']
        rfm['estimated_clv'] = rfm['monetary'] * (rfm['frequency'] / rfm['recency'].clip(lower=1)) * 365
        rfm['loyalty_score'] = (
            (1 / (rfm['recency'] + 1)) * 0.3 +
            rfm['frequency'] / rfm['frequency'].max() * 0.4 +
            rfm['monetary'] / rfm['monetary'].max() * 0.3
        ) * 100

        # Merge demographics
        if self.customers_df is not None:
            rfm = rfm.merge(self.customers_df[['customer_id', 'age_group', 'gender', 'city']], on='customer_id', how='left')

        self.rfm_df = rfm
        print(f"[RFM] Computed for {len(rfm)} customers")
        return rfm

    # ─── Segmentation ────────────────────────────────────────────────────────

    def perform_advanced_segmentation(self, method='kmeans', n_clusters_range=(3, 10)):
        """K-Means, Hierarchical, or DBSCAN clustering with optimal k selection."""
        if self.rfm_df is None:
            self.calculate_rfm_metrics()

        features = ['recency', 'frequency', 'monetary', 'avg_order_value', 'loyalty_score']
        X = self.scaler.fit_transform(self.rfm_df[features].fillna(0))

        if method == 'kmeans':
            best_score, best_k, best_labels = -1, 4, None
            for k in range(*n_clusters_range):
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = km.fit_predict(X)
                score = silhouette_score(X, labels)
                if score > best_score:
                    best_score, best_k, best_labels = score, k, labels
            self.optimal_k = best_k
            self.segment_labels = best_labels
            print(f"[SEG] KMeans: optimal k={best_k}, silhouette={best_score:.3f}")

        elif method == 'hierarchical':
            best_k = n_clusters_range[0] + 2
            model = AgglomerativeClustering(n_clusters=best_k)
            self.segment_labels = model.fit_predict(X)
            self.optimal_k = best_k

        elif method == 'dbscan':
            model = DBSCAN(eps=0.5, min_samples=5)
            self.segment_labels = model.fit_predict(X)
            self.optimal_k = len(set(self.segment_labels)) - (1 if -1 in self.segment_labels else 0)

        self.rfm_df['segment'] = self.segment_labels
        segment_names = ['VIP Champions', 'Loyal Enthusiasts', 'Big Spenders',
                         'New Promising', 'Core Customers', 'Hibernating',
                         'Growth Stars', 'Flash Buyers', 'Occasional', 'Potential']
        self.rfm_df['segment_name'] = self.rfm_df['segment'].apply(
            lambda x: segment_names[x % len(segment_names)] if x >= 0 else 'Outlier'
        )

        return self.rfm_df, self.optimal_k

    # ─── Anomaly Detection ───────────────────────────────────────────────────

    def detect_anomalies(self, contamination=0.10):
        """Isolation Forest anomaly detection."""
        if self.rfm_df is None:
            self.calculate_rfm_metrics()

        features = ['recency', 'frequency', 'monetary', 'avg_order_value']
        X = self.rfm_df[features].fillna(0)
        iso = IsolationForest(contamination=contamination, random_state=42)
        self.rfm_df['anomaly_score'] = iso.fit_predict(X)
        self.rfm_df['anomaly_confidence'] = -iso.score_samples(X)

        self.anomalies_df = self.rfm_df[self.rfm_df['anomaly_score'] == -1].copy()
        print(f"[ANOMALY] Detected {len(self.anomalies_df)} anomalous customers ({contamination*100:.0f}% rate)")
        return self.anomalies_df

    # ─── Segment Analysis ────────────────────────────────────────────────────

    def analyze_segments(self):
        """Compute per-segment KPIs."""
        if 'segment_name' not in self.rfm_df.columns:
            self.perform_advanced_segmentation()

        analysis = self.rfm_df.groupby('segment_name').agg(
            count=('customer_id', 'count'),
            avg_recency=('recency', 'mean'),
            avg_frequency=('frequency', 'mean'),
            avg_monetary=('monetary', 'mean'),
            total_clv=('estimated_clv', 'sum'),
            avg_loyalty=('loyalty_score', 'mean'),
        ).reset_index()
        analysis['pct_customers'] = analysis['count'] / analysis['count'].sum() * 100
        return analysis

    # ─── Business Insights ───────────────────────────────────────────────────

    def generate_business_insights(self):
        """Auto-compute key business KPIs."""
        if self.rfm_df is None:
            self.calculate_rfm_metrics()

        total_revenue = self.transactions_df['amount'].sum()
        at_risk = self.rfm_df[self.rfm_df['recency'] > 180]
        high_value = self.rfm_df.nlargest(int(len(self.rfm_df) * 0.2), 'estimated_clv')

        return {
            'total_customers': len(self.rfm_df),
            'total_revenue': round(total_revenue, 2),
            'avg_order_value': round(self.transactions_df['amount'].mean(), 2),
            'total_clv': round(self.rfm_df['estimated_clv'].sum(), 2),
            'avg_clv': round(self.rfm_df['estimated_clv'].mean(), 2),
            'at_risk_customers': len(at_risk),
            'high_value_customers': len(high_value),
            'category_revenue': self.transactions_df.groupby('category')['amount'].sum().to_dict(),
            'channel_revenue': self.transactions_df.groupby('channel')['amount'].sum().to_dict(),
        }

    # ─── Dashboard ───────────────────────────────────────────────────────────

    def create_comprehensive_dashboard(self, export_html: str = None):
        """Build multi-panel interactive Plotly dashboard."""
        if 'segment_name' not in self.rfm_df.columns:
            self.perform_advanced_segmentation()

        seg_counts = self.rfm_df['segment_name'].value_counts().reset_index()
        seg_counts.columns = ['Segment', 'Count']

        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=[
                'Customer Segment Distribution', '3D RFM Scatter',
                'CLV by Segment', 'Revenue by Category',
                'Channel Performance', 'Loyalty Score Distribution',
                'Recency vs Monetary', 'Anomaly Detection (PCA)',
                'Monthly Revenue Trend'
            ],
            specs=[
                [{'type': 'pie'}, {'type': 'scatter3d'}, {'type': 'bar'}],
                [{'type': 'bar'}, {'type': 'bar'}, {'type': 'histogram'}],
                [{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}],
            ]
        )

        # 1 - Pie
        fig.add_trace(go.Pie(labels=seg_counts['Segment'], values=seg_counts['Count'], hole=0.4), row=1, col=1)

        # 2 - 3D Scatter
        pca = PCA(n_components=3)
        coords = pca.fit_transform(self.scaler.transform(
            self.rfm_df[['recency', 'frequency', 'monetary', 'avg_order_value', 'loyalty_score']].fillna(0)
        ))
        fig.add_trace(go.Scatter3d(
            x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
            mode='markers',
            marker=dict(size=3, color=self.rfm_df['segment'], colorscale='Viridis', opacity=0.7),
        ), row=1, col=2)

        # 3 - CLV by segment
        seg_clv = self.rfm_df.groupby('segment_name')['estimated_clv'].mean().reset_index()
        fig.add_trace(go.Bar(x=seg_clv['segment_name'], y=seg_clv['estimated_clv'],
                             marker_color='#6366f1'), row=1, col=3)

        # 4 - Category revenue
        cat_rev = self.transactions_df.groupby('category')['amount'].sum().reset_index()
        fig.add_trace(go.Bar(x=cat_rev['category'], y=cat_rev['amount'],
                             marker_color='#22d3ee'), row=2, col=1)

        # 5 - Channel revenue
        ch_rev = self.transactions_df.groupby('channel')['amount'].sum().reset_index()
        fig.add_trace(go.Bar(x=ch_rev['channel'], y=ch_rev['amount'],
                             marker_color='#f59e0b'), row=2, col=2)

        # 6 - Loyalty histogram
        fig.add_trace(go.Histogram(x=self.rfm_df['loyalty_score'], nbinsx=30,
                                   marker_color='#ec4899'), row=2, col=3)

        # 7 - Recency vs Monetary
        fig.add_trace(go.Scatter(x=self.rfm_df['recency'], y=self.rfm_df['monetary'],
                                 mode='markers', marker=dict(size=4, opacity=0.5, color='#34d399')), row=3, col=1)

        # 8 - Anomaly PCA
        pca2 = PCA(n_components=2)
        xy = pca2.fit_transform(self.rfm_df[['recency', 'frequency', 'monetary', 'avg_order_value']].fillna(0))
        colors = ['red' if s == -1 else '#6366f1' for s in self.rfm_df.get('anomaly_score', [1] * len(self.rfm_df))]
        fig.add_trace(go.Scatter(x=xy[:, 0], y=xy[:, 1], mode='markers',
                                 marker=dict(size=4, color=colors, opacity=0.6)), row=3, col=2)

        # 9 - Monthly revenue trend
        monthly = self.transactions_df.copy()
        monthly['month'] = monthly['transaction_date'].dt.to_period('M').astype(str)
        mrev = monthly.groupby('month')['amount'].sum().reset_index().sort_values('month')
        fig.add_trace(go.Scatter(x=mrev['month'], y=mrev['amount'],
                                 mode='lines+markers', line=dict(color='#f97316', width=2)), row=3, col=3)

        fig.update_layout(
            height=1200, showlegend=False,
            title_text="AI Retail Intelligence Platform V3.0 - Analytics Dashboard",
            title_font_size=18,
            paper_bgcolor='#0f172a', plot_bgcolor='#1e293b',
            font=dict(color='#e2e8f0'),
        )

        if export_html:
            fig.write_html(export_html)
            print(f"[EXPORT] Dashboard saved -> {export_html}")

        return fig

    # ─── Report ──────────────────────────────────────────────────────────────

    def generate_comprehensive_report(self):
        """Print a structured text report."""
        insights = self.generate_business_insights()
        seg_analysis = self.analyze_segments()

        SEP = "=" * 65
        print("\n" + SEP)
        print("  AI RETAIL INTELLIGENCE PLATFORM V3.0 - REPORT")
        print(SEP)
        print(f"\n  BUSINESS OVERVIEW (Store: {self.store_id})")
        print(f"   Total Customers     : {insights['total_customers']:,}")
        print(f"   Total Revenue       : ${insights['total_revenue']:,.2f}")
        print(f"   Avg Order Value     : ${insights['avg_order_value']:,.2f}")
        print(f"   Total CLV           : ${insights['total_clv']:,.2f}")
        print(f"   At-Risk Customers   : {insights['at_risk_customers']:,}")
        print(f"   High-Value Cust.    : {insights['high_value_customers']:,}")

        print(f"\n  CUSTOMER SEGMENTS ({self.optimal_k} clusters)")
        for _, row in seg_analysis.iterrows():
            print(f"   - {row['segment_name']}: {row['count']} ({row['pct_customers']:.1f}%)")

        print("\n  CATEGORY REVENUE")
        for cat, rev in sorted(insights['category_revenue'].items(), key=lambda x: -x[1]):
            print(f"   - {cat}: ${rev:,.2f}")

        print("\n  CHANNEL REVENUE")
        for ch, rev in sorted(insights['channel_revenue'].items(), key=lambda x: -x[1]):
            print(f"   - {ch}: ${rev:,.2f}")
        print(SEP)

        return {**insights, 'segments': seg_analysis.to_dict('records')}


if __name__ == "__main__":
    rip = RetailIntelligencePlatform()
    rip.generate_synthetic_data(n_customers=1000, n_transactions=5000)
    rip.calculate_rfm_metrics()
    rip.perform_advanced_segmentation()
    rip.detect_anomalies()
    rip.analyze_segments()
    rip.generate_comprehensive_report()
    fig = rip.create_comprehensive_dashboard(export_html="exports/dashboard.html")
    print("\n[DONE] Platform run complete. Open exports/dashboard.html to view.")
