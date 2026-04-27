import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Sparkles } from 'lucide-react';
import { SEGMENT_META } from '../utils/rfm';
import { getSegmentInsight } from '../utils/gemini';
import { segCardHover } from '../motion/variants';

function StatPill({ label, value, color }) {
  return (
    <div style={{ textAlign: 'center' }}>
      <div className="seg-stat-label">{label}</div>
      <div className="seg-stat-val" style={{ color }}>{value}</div>
    </div>
  );
}

export default function SegmentCard({ seg, kpis, geminiKey }) {
  const [insight, setInsight] = useState(null);
  const [loading, setLoading] = useState(false);
  const meta = SEGMENT_META[seg.segment] ?? { color: '#6366f1', icon: '🎯', description: '' };

  useEffect(() => {
    let cancelled = false;
    setInsight(null);
    setLoading(true);
    getSegmentInsight(seg, kpis, geminiKey).then(r => {
      if (!cancelled) { setInsight(r); setLoading(false); }
    });
    return () => { cancelled = true; };
  }, [seg.segment, geminiKey]);

  const pct = parseFloat(seg.pct);

  return (
    <motion.div className="segment-card" whileHover={segCardHover} style={{ cursor: 'default' }}>
      <div className="segment-card-accent" style={{ background: `linear-gradient(90deg, ${meta.color}, transparent)` }} />

      <div className="seg-header">
        <div className="seg-title">
          <span className="seg-icon">{meta.icon}</span>
          <span className="seg-name">{seg.segment}</span>
        </div>
        <span className="seg-count">{seg.count} · {seg.pct}%</span>
      </div>

      {/* Progress bar */}
      <div className="progress-bar" style={{ marginBottom: '1rem' }}>
        <div className="progress-fill" style={{ width: `${Math.min(pct * 2, 100)}%`, background: meta.color }} />
      </div>

      <div className="seg-stats">
        <StatPill label="Recency"   value={`${seg.avg_recency}d`}   color={meta.color} />
        <StatPill label="Frequency" value={seg.avg_frequency}        color={meta.color} />
        <StatPill label="Avg Spend" value={`$${seg.avg_monetary}`}   color={meta.color} />
      </div>

      <div style={{ display: 'flex', gap: '1rem', marginBottom: '1rem' }}>
        <div style={{ flex: 1 }}>
          <div className="seg-stat-label">Avg CLV</div>
          <div className="seg-stat-val" style={{ color: meta.color, fontSize: '1rem' }}>${Math.round(seg.avg_clv).toLocaleString()}</div>
        </div>
        <div style={{ flex: 1 }}>
          <div className="seg-stat-label">Revenue</div>
          <div className="seg-stat-val" style={{ color: meta.color, fontSize: '1rem' }}>${Math.round(seg.total_revenue).toLocaleString()}</div>
        </div>
        <div style={{ flex: 1 }}>
          <div className="seg-stat-label">Loyalty</div>
          <div className="seg-stat-val" style={{ color: meta.color, fontSize: '1rem' }}>{seg.avg_loyalty}/100</div>
        </div>
      </div>

      <div className="seg-description">{meta.description}</div>

      {/* Gemini Insight */}
      <div className="insight-box">
        <div className="insight-header">
          <Sparkles size={11} />
          AI Insight {geminiKey ? '· Gemini' : '· Smart Rules'}
        </div>

        {loading && (
          <div className="insight-loading">
            <div className="spinner" />
            <span>Generating insight...</span>
          </div>
        )}

        {insight && !loading && (
          <>
            <div className="insight-summary">{insight.summary}</div>
            {insight.opportunity && (
              <div style={{ fontSize: '.75rem', color: 'var(--accent2)', marginBottom: '.6rem', fontStyle: 'italic' }}>
                💡 {insight.opportunity}
              </div>
            )}
            <div className="insight-actions">
              {insight.actions?.map((a, i) => (
                <div key={i} className="insight-action">{a}</div>
              ))}
            </div>
            {insight.priority && (
              <div className={`insight-priority priority-${insight.priority}`}>
                {insight.priority} PRIORITY
              </div>
            )}
          </>
        )}
      </div>
    </motion.div>
  );
}
