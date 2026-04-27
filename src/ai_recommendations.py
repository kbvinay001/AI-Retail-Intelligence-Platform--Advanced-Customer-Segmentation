"""
V3.0 — AI Recommendations Engine
GPT-4 powered business strategy suggestions with fallback rule-based engine.
"""

import os
import json
from typing import Optional, Dict, List
from datetime import datetime

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# ─── Rule-Based Recommendation Engine (always available) ─────────────────────

RULE_TEMPLATES = {
    'high_churn_risk': {
        'trigger': lambda kpis: kpis.get('at_risk_customers', 0) / max(kpis.get('total_customers', 1), 1) > 0.25,
        'title': '🚨 High Churn Risk Detected',
        'recommendation': (
            "Over 25% of your customers haven't purchased in 180+ days. "
            "Launch a win-back campaign: personalized email sequences with a 15-20% discount "
            "targeted at high-CLV dormant customers. Set up automated triggers at 90/120/180-day "
            "inactivity thresholds."
        ),
        'priority': 'CRITICAL',
        'actions': [
            'Segment at-risk customers by historical CLV',
            'Create tiered win-back offers (15% / 20% / 25%)',
            'Launch email drip sequence over 3 weeks',
            'Monitor re-activation rate weekly',
        ]
    },
    'low_aov': {
        'trigger': lambda kpis: kpis.get('avg_order_value', 999) < 50,
        'title': '📦 Low Average Order Value',
        'recommendation': (
            "Your AOV is below $50. Implement bundle recommendations and free-shipping thresholds "
            "($75+) to encourage larger basket sizes. Cross-category upsell campaigns can increase "
            "AOV by 20-35%."
        ),
        'priority': 'HIGH',
        'actions': [
            'Add product bundle recommendations at checkout',
            'Set free shipping threshold at 1.5× current AOV',
            'Launch cross-category promotion for top segments',
            'A/B test upsell widget placement',
        ]
    },
    'strong_vip_segment': {
        'trigger': lambda kpis: kpis.get('high_value_customers', 0) / max(kpis.get('total_customers', 1), 1) > 0.15,
        'title': '💎 Strong VIP Segment Opportunity',
        'recommendation': (
            "15%+ of your customers are high-value. Launch an exclusive loyalty tier with "
            "early access, dedicated support, and premium rewards. VIP programs typically "
            "increase retention by 30% and CLV by 45%."
        ),
        'priority': 'HIGH',
        'actions': [
            'Define VIP tier criteria (top 15% by CLV)',
            'Design exclusive benefits: early access, free shipping, birthday bonus',
            'Send personalized VIP invitation sequence',
            'Track NPS and repeat purchase rate for VIPs',
        ]
    },
    'high_clv': {
        'trigger': lambda kpis: kpis.get('avg_clv', 0) > 500,
        'title': '🚀 High CLV Base — Expand Market',
        'recommendation': (
            "Your average CLV exceeds $500, indicating a high-quality customer base. "
            "Accelerate acquisition by investing in referral programs and lookalike audience "
            "campaigns targeting customers with similar profiles."
        ),
        'priority': 'MEDIUM',
        'actions': [
            'Launch referral program with $25 credit incentive',
            'Build lookalike audience from top 10% CLV customers',
            'Increase paid acquisition budget by 20%',
            'Track CAC vs CLV ratio monthly',
        ]
    },
    'category_concentration': {
        'trigger': lambda kpis: True,  # Always show category advice
        'title': '🛍️ Optimize Category Mix',
        'recommendation': (
            "Analyze under-performing categories and cross-sell opportunities. "
            "Customers who buy across 3+ categories have 4× higher retention rates. "
            "Introduce category discovery campaigns for single-category buyers."
        ),
        'priority': 'MEDIUM',
        'actions': [
            'Identify single-category customers (>60% of base)',
            'Create cross-category starter bundles',
            'Test category recommendation emails',
            'Track category diversification rate quarterly',
        ]
    },
}


class AIRecommendationsEngine:
    """
    Generates strategic retail recommendations using:
    1. GPT-4 (if OPENAI_API_KEY is set)
    2. Rule-based engine (always available as fallback)
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.use_gpt = OPENAI_AVAILABLE and bool(self.api_key)
        self._client = None

        if self.use_gpt:
            openai.api_key = self.api_key
            self._client = openai.OpenAI(api_key=self.api_key)
            print("[AI] Recommendations: GPT-4 mode enabled")
        else:
            print("[AI] Recommendations: Rule-based mode (set OPENAI_API_KEY for GPT-4)")

    # ─── Rule-Based Engine ────────────────────────────────────────────────────

    def _rule_based_recommendations(self, kpis: dict, segment_analysis: list) -> List[dict]:
        recs = []
        for rule_id, rule in RULE_TEMPLATES.items():
            if rule['trigger'](kpis):
                recs.append({
                    'id': rule_id,
                    'title': rule['title'],
                    'recommendation': rule['recommendation'],
                    'priority': rule['priority'],
                    'actions': rule['actions'],
                    'source': 'rule_engine',
                    'generated_at': datetime.now().isoformat(),
                })
        return recs

    # ─── GPT-4 Engine ────────────────────────────────────────────────────────

    def _gpt_recommendations(self, kpis: dict, segment_analysis: list) -> List[dict]:
        system_prompt = (
            "You are a senior retail analytics consultant. "
            "Analyze the provided customer KPIs and segmentation data, "
            "then return 5 actionable business recommendations as a JSON array. "
            "Each object must have: title, recommendation, priority (CRITICAL/HIGH/MEDIUM/LOW), actions (list of 4 strings)."
        )
        user_prompt = (
            f"Business KPIs:\n{json.dumps(kpis, indent=2)}\n\n"
            f"Segment Analysis:\n{json.dumps(segment_analysis[:6], indent=2)}\n\n"
            "Return ONLY valid JSON array, no markdown."
        )

        try:
            response = self._client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0.4,
                max_tokens=2000,
            )
            raw = response.choices[0].message.content.strip()
            recs = json.loads(raw)
            for r in recs:
                r['source'] = 'gpt-4'
                r['generated_at'] = datetime.now().isoformat()
                r.setdefault('id', r.get('title', 'gpt_rec').lower().replace(' ', '_'))
            print(f"[OK] GPT-4 generated {len(recs)} recommendations")
            return recs
        except Exception as e:
            print(f"[WARN] GPT-4 error ({e}). Falling back to rule engine.")
            return self._rule_based_recommendations(kpis, segment_analysis)

    # ─── Public API ──────────────────────────────────────────────────────────

    def generate(self, kpis: dict, segment_analysis: list) -> List[dict]:
        """Generate recommendations. Uses GPT-4 if available, else rule engine."""
        if self.use_gpt:
            return self._gpt_recommendations(kpis, segment_analysis)
        return self._rule_based_recommendations(kpis, segment_analysis)

    def print_recommendations(self, recommendations: List[dict]):
        """Pretty-print recommendations to console."""
        priority_icons = {'CRITICAL': '[!!!]', 'HIGH': '[!!]', 'MEDIUM': '[!]', 'LOW': '[i]'}
        print("\n" + "=" * 60)
        print("  AI-POWERED BUSINESS RECOMMENDATIONS")
        print("=" * 60)
        for i, rec in enumerate(recommendations, 1):
            icon = priority_icons.get(rec.get('priority', 'MEDIUM'), '[!]')
            print(f"\n{i}. {icon} [{rec.get('priority','MEDIUM')}] {rec['title']}")
            print(f"   {rec['recommendation']}")
            print(f"   Action Plan:")
            for action in rec.get('actions', []):
                print(f"      - {action}")
        print("=" * 60)
