/**
 * RFM Scoring + Segmentation Engine (JavaScript port)
 * Recency · Frequency · Monetary
 */

/** Compute quintile rank (1-5) for an array of values */
function quintileScore(values, ascending = true) {
  const sorted = [...values].sort((a, b) => a - b);
  const n = sorted.length;
  return values.map(v => {
    const rank = sorted.filter(x => x <= v).length / n; // 0-1 percentile
    const score = Math.min(5, Math.ceil(rank * 5));
    return ascending ? score : 6 - score; // invert for recency (lower days = better)
  });
}

/** Compute RFM metrics from transactions array */
export function computeRFM(transactions) {
  const now = new Date();
  // Group by customer
  const byCustomer = {};
  for (const t of transactions) {
    const cid = String(t.customer_id);
    if (!byCustomer[cid]) byCustomer[cid] = { dates: [], amounts: [], txnIds: new Set() };
    const date = new Date(t.transaction_date);
    if (!isNaN(date)) byCustomer[cid].dates.push(date);
    byCustomer[cid].amounts.push(t.amount);
    byCustomer[cid].txnIds.add(t.transaction_id);
  }

  const records = Object.entries(byCustomer).map(([cid, data]) => {
    const lastDate   = new Date(Math.max(...data.dates));
    const recency    = Math.round((now - lastDate) / (1000 * 60 * 60 * 24)); // days
    const frequency  = data.txnIds.size;
    const monetary   = data.amounts.reduce((a, b) => a + b, 0);
    const aov        = monetary / frequency;
    const tenureDays = data.dates.length > 1
      ? Math.round((Math.max(...data.dates) - Math.min(...data.dates)) / (1000 * 60 * 60 * 24))
      : 0;
    const clv = monetary * (frequency / Math.max(recency, 1)) * 365;
    return { customer_id: cid, recency, frequency, monetary, aov, clv, tenureDays };
  });

  if (records.length === 0) return [];

  // Quintile scores
  const rScores = quintileScore(records.map(r => r.recency), false); // lower recency = better
  const fScores = quintileScore(records.map(r => r.frequency), true);
  const mScores = quintileScore(records.map(r => r.monetary), true);

  return records.map((r, i) => ({
    ...r,
    r_score:      rScores[i],
    f_score:      fScores[i],
    m_score:      mScores[i],
    rfm_score:    rScores[i] * 100 + fScores[i] * 10 + mScores[i],
    loyalty_score: Math.round(
      (rScores[i] / 5) * 0.3 * 100 +
      (fScores[i] / 5) * 0.4 * 100 +
      (mScores[i] / 5) * 0.3 * 100
    ),
    segment: classifySegment(rScores[i], fScores[i], mScores[i]),
  }));
}

/** Rule-based RFM segmentation */
function classifySegment(r, f, m) {
  if (r >= 4 && f >= 4 && m >= 4) return 'VIP Champions';
  if (r >= 3 && f >= 4)            return 'Loyal Enthusiasts';
  if (r >= 4 && f <= 1)            return 'New Promising';
  if (m >= 4 && r >= 3)            return 'Big Spenders';
  if (r <= 2 && f >= 3)            return 'At Risk';
  if (r <= 1)                       return 'Hibernating';
  if (r >= 3 && f <= 2)            return 'Potential Loyalists';
  return 'Core Customers';
}

export const SEGMENT_META = {
  'VIP Champions':       { color: '#a855f7', icon: '💎', description: 'Best customers — buy often, spend most, bought recently.' },
  'Loyal Enthusiasts':   { color: '#6366f1', icon: '🔥', description: 'Regular buyers with strong frequency. High retention.' },
  'New Promising':       { color: '#22d3ee', icon: '🌱', description: 'Recent customers who just started. High potential.' },
  'Big Spenders':        { color: '#f59e0b', icon: '💰', description: 'High monetary value even if not frequent.' },
  'At Risk':             { color: '#ef4444', icon: '⚠️',  description: 'Were frequent buyers — now going dormant. Act fast.' },
  'Hibernating':         { color: '#64748b', icon: '😴', description: 'Lost customers. Very low recency and frequency.' },
  'Potential Loyalists': { color: '#34d399', icon: '🚀', description: 'Recent customers who could become loyal with nurturing.' },
  'Core Customers':      { color: '#f97316', icon: '🎯', description: 'Average across all RFM dimensions. Solid base.' },
};

/** Aggregate per-segment KPIs */
export function aggregateSegments(rfmData) {
  const groups = {};
  for (const r of rfmData) {
    if (!groups[r.segment]) groups[r.segment] = [];
    groups[r.segment].push(r);
  }
  return Object.entries(groups).map(([segment, members]) => ({
    segment,
    count:        members.length,
    pct:          (members.length / rfmData.length * 100).toFixed(1),
    avg_recency:  Math.round(members.reduce((s, m) => s + m.recency, 0) / members.length),
    avg_frequency: +(members.reduce((s, m) => s + m.frequency, 0) / members.length).toFixed(1),
    avg_monetary: +(members.reduce((s, m) => s + m.monetary, 0) / members.length).toFixed(2),
    avg_clv:      +(members.reduce((s, m) => s + m.clv, 0) / members.length).toFixed(2),
    avg_loyalty:  Math.round(members.reduce((s, m) => s + m.loyalty_score, 0) / members.length),
    total_revenue: +(members.reduce((s, m) => s + m.monetary, 0)).toFixed(2),
    color:        SEGMENT_META[segment]?.color ?? '#6366f1',
  })).sort((a, b) => b.total_revenue - a.total_revenue);
}

/** Top-level KPIs */
export function computeKPIs(rfmData, transactions) {
  const totalRevenue  = transactions.reduce((s, t) => s + t.amount, 0);
  const totalCLV      = rfmData.reduce((s, r) => s + r.clv, 0);
  const atRisk        = rfmData.filter(r => r.recency > 180).length;
  const highValue     = rfmData.filter(r => r.clv >= rfmData.map(x => x.clv).sort((a,b)=>b-a)[Math.floor(rfmData.length*0.2)]).length;
  const catRevenue    = {};
  const chanRevenue   = {};
  for (const t of transactions) {
    catRevenue[t.category]  = (catRevenue[t.category]  ?? 0) + t.amount;
    chanRevenue[t.channel]  = (chanRevenue[t.channel]  ?? 0) + t.amount;
  }
  return {
    totalCustomers: rfmData.length,
    totalRevenue:   +totalRevenue.toFixed(2),
    avgOrderValue:  +(totalRevenue / transactions.length).toFixed(2),
    totalCLV:       +totalCLV.toFixed(2),
    avgCLV:         +(totalCLV / rfmData.length).toFixed(2),
    atRiskCount:    atRisk,
    highValueCount: highValue,
    categoryRevenue: Object.entries(catRevenue).sort((a,b)=>b[1]-a[1]),
    channelRevenue:  Object.entries(chanRevenue).sort((a,b)=>b[1]-a[1]),
  };
}

/** Monthly revenue trend from transactions */
export function monthlyRevenue(transactions) {
  const monthly = {};
  for (const t of transactions) {
    const m = t.transaction_date?.slice(0, 7) ?? '2024-01';
    monthly[m] = (monthly[m] ?? 0) + t.amount;
  }
  return Object.entries(monthly)
    .sort((a, b) => a[0].localeCompare(b[0]))
    .map(([month, revenue]) => ({ month, revenue: +revenue.toFixed(2) }));
}
