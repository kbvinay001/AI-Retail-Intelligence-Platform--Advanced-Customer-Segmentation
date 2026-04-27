"""
AI Retail Intelligence Platform V3.0 - Complete Demo Runner
Runs the full pipeline: data generation -> RFM -> segmentation ->
anomaly detection -> forecasting -> AI recommendations -> dashboard export.
"""
import sys, io, os
# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from retail_intelligence import RetailIntelligencePlatform
from forecasting_engine import ForecastingEngine
from ai_recommendations import AIRecommendationsEngine
from multi_store import MultiStoreAnalytics, StoreConfig

SEP = "=" * 62

BANNER = f"""
{SEP}
  AI RETAIL INTELLIGENCE PLATFORM  -  VERSION 3.0
  Enterprise Analytics | Multi-store | REST API | GPT-AI
{SEP}
"""


def run_single_store_demo():
    print(f"\n{SEP}")
    print("  DEMO 1: Single-Store Full Pipeline")
    print(SEP)

    rip = RetailIntelligencePlatform(store_id="FLAGSHIP_001")
    rip.generate_synthetic_data(n_customers=1000, n_transactions=5000)
    rip.calculate_rfm_metrics()
    rip.perform_advanced_segmentation(method='kmeans')
    rip.detect_anomalies(contamination=0.10)
    report = rip.generate_comprehensive_report()

    os.makedirs("exports", exist_ok=True)
    fig = rip.create_comprehensive_dashboard(export_html="exports/dashboard_v3.html")
    print("\n[OK] Dashboard saved -> exports/dashboard_v3.html")
    return rip, report


def run_forecasting_demo(rip):
    print(f"\n{SEP}")
    print("  DEMO 2: Advanced Revenue Forecasting")
    print(SEP)

    eng = ForecastingEngine(model='simple')
    fc = eng.forecast(rip.transactions_df, horizon=90)
    summary = eng.get_forecast_summary()

    print("\n90-Day Forecast Summary:")
    print(f"   Total projected revenue : ${summary['total_projected_revenue']:,.2f}")
    print(f"   Average daily revenue   : ${summary['avg_daily_revenue']:,.2f}")
    print(f"   Peak day                : {summary['peak_day']}")
    print(f"   Peak revenue            : ${summary['peak_revenue']:,.2f}")
    print(f"   Growth trend            : {summary['growth_trend_pct']:+.1f}%")
    return fc, summary


def run_ai_recommendations_demo(rip):
    print(f"\n{SEP}")
    print("  DEMO 3: AI-Powered Business Recommendations")
    print(SEP)

    engine = AIRecommendationsEngine()
    kpis = rip.generate_business_insights()
    seg = rip.analyze_segments().to_dict('records')
    recs = engine.generate(kpis, seg)
    engine.print_recommendations(recs)
    return recs


def run_multi_store_demo():
    print(f"\n{SEP}")
    print("  DEMO 4: Multi-Store Network Analytics")
    print(SEP)

    ms = MultiStoreAnalytics(tenant_id="ENTERPRISE_001")
    store_configs = [
        StoreConfig("MUM_001", "Mumbai Flagship",   "Mumbai",    "West",  "flagship"),
        StoreConfig("DEL_001", "Delhi Central",     "Delhi",     "North", "standard"),
        StoreConfig("BLR_001", "Bangalore Tech Hub","Bangalore", "South", "flagship"),
        StoreConfig("CHN_001", "Chennai Express",   "Chennai",   "South", "express"),
        StoreConfig("HYD_001", "Hyderabad Pearl",   "Hyderabad", "South", "standard"),
    ]
    for cfg in store_configs:
        ms.register_store(cfg)

    ms.populate_all_stores(customers_per_store=400, txn_per_store=1500)
    ms.print_network_summary()
    return ms


def main():
    print(BANNER)
    rip, report   = run_single_store_demo()
    fc, fc_summary = run_forecasting_demo(rip)
    recs           = run_ai_recommendations_demo(rip)
    ms             = run_multi_store_demo()

    print(f"\n{SEP}")
    print("  ALL DEMOS COMPLETE")
    print(SEP)
    print("  [Dashboard] -> exports/dashboard_v3.html")
    print("  [REST API]  -> python src/api.py  (http://localhost:8000/docs)")
    print("  [Tests]     -> python -m pytest tests/ -v")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()
