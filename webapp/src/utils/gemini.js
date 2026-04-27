/**
 * Gemini API Integration
 * Generates per-segment strategic insights using Google Gemini.
 */
import { GoogleGenerativeAI } from '@google/generative-ai';

let _client = null;

export function initGemini(apiKey) {
  if (!apiKey) return null;
  _client = new GoogleGenerativeAI(apiKey);
  return _client;
}

export async function getSegmentInsight(segmentData, kpis, apiKey) {
  if (!apiKey) return getFallbackInsight(segmentData);

  try {
    if (!_client) initGemini(apiKey);
    const model = _client.getGenerativeModel({ model: 'gemini-2.5-flash' });

    const prompt = `You are a senior retail analytics consultant. Analyze this customer segment and provide 3 concise, actionable strategic recommendations.

Segment: ${segmentData.segment}
Customers: ${segmentData.count} (${segmentData.pct}% of base)
Avg Recency: ${segmentData.avg_recency} days since last purchase
Avg Frequency: ${segmentData.avg_frequency} purchases
Avg Monetary: $${segmentData.avg_monetary}
Avg CLV: $${segmentData.avg_clv}
Avg Loyalty Score: ${segmentData.avg_loyalty}/100
Total Revenue from segment: $${segmentData.total_revenue}

Overall Business KPIs:
- Total Customers: ${kpis.totalCustomers}
- Total Revenue: $${kpis.totalRevenue}
- Avg Order Value: $${kpis.avgOrderValue}

Respond in JSON format only:
{
  "summary": "One sentence strategic summary of this segment",
  "opportunity": "Key opportunity or risk in one sentence",
  "actions": ["Action 1", "Action 2", "Action 3"],
  "priority": "HIGH | MEDIUM | LOW"
}`;

    const result = await model.generateContent(prompt);
    const text = result.response.text().trim();
    const jsonMatch = text.match(/\{[\s\S]*\}/);
    if (jsonMatch) return JSON.parse(jsonMatch[0]);
    return getFallbackInsight(segmentData);
  } catch (err) {
    console.warn('Gemini API error:', err.message);
    return getFallbackInsight(segmentData);
  }
}

const FALLBACK_INSIGHTS = {
  'VIP Champions': {
    summary: 'Your most valuable segment — protect and reward them relentlessly.',
    opportunity: 'VIP loyalty program with exclusive perks can increase CLV by 40-50%.',
    actions: [
      'Launch an invite-only VIP tier with early access to new products',
      'Assign dedicated account managers for top 20 VIP customers',
      'Send personalized anniversary and birthday rewards quarterly',
    ],
    priority: 'HIGH',
  },
  'Loyal Enthusiasts': {
    summary: 'High-frequency buyers with strong brand affinity and retention.',
    opportunity: 'Upselling and cross-category promotion can increase their monetary value by 25%.',
    actions: [
      'Introduce product bundles targeting their purchase history',
      'Offer double loyalty points for cross-category purchases',
      'Create referral program with $25 credit for each new customer they bring',
    ],
    priority: 'HIGH',
  },
  'New Promising': {
    summary: 'Recent acquirees with high potential if onboarded correctly.',
    opportunity: 'Strong onboarding sequence in first 30 days increases 90-day retention by 60%.',
    actions: [
      'Send welcome sequence: day 3, 7, 14 post-purchase emails',
      'Offer 10% off second purchase to reduce friction',
      'Showcase top products in their first-purchase category',
    ],
    priority: 'HIGH',
  },
  'Big Spenders': {
    summary: 'High monetary value even with infrequent purchases — premium audience.',
    opportunity: 'Increasing visit frequency by even 1 additional purchase/year can double revenue.',
    actions: [
      'Launch curated "Premium Collection" newsletter monthly',
      'Offer VIP preview events for new collections',
      'Introduce subscription or replenishment model for repeat items',
    ],
    priority: 'MEDIUM',
  },
  'At Risk': {
    summary: 'Formerly active customers now showing signs of churn — act immediately.',
    opportunity: 'Win-back campaigns recover 15-25% of at-risk customers at low acquisition cost.',
    actions: [
      'Launch automated 3-email win-back sequence with 15% discount',
      'Conduct 5-customer exit survey to identify churn reasons',
      'Retarget via paid ads using their historical category preferences',
    ],
    priority: 'HIGH',
  },
  'Hibernating': {
    summary: 'Long-dormant customers with low re-engagement probability.',
    opportunity: 'Sunset campaign recovers ~10% while reducing list costs by removing the rest.',
    actions: [
      'Send a final "We miss you" email with 25% coupon — one last attempt',
      'Suppress unresponsive customers from regular campaigns after 90 days',
      'Analyze common exit categories to fix product-market fit issues',
    ],
    priority: 'MEDIUM',
  },
  'Potential Loyalists': {
    summary: 'Recent buyers who need nurturing to convert to loyal customers.',
    opportunity: 'Targeted engagement in next 30 days can move 40% into Loyal Enthusiasts.',
    actions: [
      'Enroll in onboarding loyalty program with points for next purchase',
      'Send "Top picks for you" personalized recommendations weekly',
      'Offer free shipping on next 3 orders to encourage frequency',
    ],
    priority: 'MEDIUM',
  },
  'Core Customers': {
    summary: 'Average across all RFM dimensions — your reliable revenue base.',
    opportunity: 'Small improvements in frequency or basket size yield outsized revenue gains.',
    actions: [
      'Launch free-shipping threshold at 1.5x their current average order value',
      'A/B test product recommendation widgets at checkout',
      'Send monthly "Best Sellers" digest to increase visit frequency',
    ],
    priority: 'LOW',
  },
};

function getFallbackInsight(segmentData) {
  return FALLBACK_INSIGHTS[segmentData.segment] ?? {
    summary: `The ${segmentData.segment} segment requires focused attention.`,
    opportunity: 'Targeted campaigns based on RFM patterns can improve performance.',
    actions: [
      'Analyze purchase patterns and identify top products',
      'Create segment-specific email campaign',
      'Track NPS and repeat purchase rate monthly',
    ],
    priority: 'MEDIUM',
  };
}
