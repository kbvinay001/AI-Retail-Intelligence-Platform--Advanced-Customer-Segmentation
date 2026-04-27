import {
  PieChart, Pie, Cell, Tooltip, ResponsiveContainer, Legend,
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  ScatterChart, Scatter, ZAxis,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  LineChart, Line, AreaChart, Area,
} from 'recharts';
import { SEGMENT_META } from '../utils/rfm';

const CHART_COLORS = ['#6366f1','#22d3ee','#a855f7','#f59e0b','#ef4444','#34d399','#f97316','#64748b'];

// ─── Custom Tooltip ──────────────────────────────────────────────────────────
const CustomTooltip = ({ active, payload, label, prefix = '', suffix = '' }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: '#1a1a2e', border: '1px solid #2a2a45',
      borderRadius: 10, padding: '10px 14px', fontSize: '.78rem',
    }}>
      {label && <div style={{ color: '#94a3b8', marginBottom: 4 }}>{label}</div>}
      {payload.map((p, i) => (
        <div key={i} style={{ color: p.color || '#e2e8f0', fontWeight: 600 }}>
          {p.name}: {prefix}{typeof p.value === 'number' ? p.value.toLocaleString(undefined, { maximumFractionDigits: 2 }) : p.value}{suffix}
        </div>
      ))}
    </div>
  );
};

// ─── Segment Distribution Pie ────────────────────────────────────────────────
export function SegmentPieChart({ segments }) {
  return (
    <div className="chart-card">
      <div className="chart-card-title">Customer Segment Distribution</div>
      <ResponsiveContainer width="100%" height={260}>
        <PieChart>
          <Pie data={segments} dataKey="count" nameKey="segment" cx="50%" cy="50%"
            innerRadius={55} outerRadius={95} paddingAngle={3}>
            {segments.map((s, i) => (
              <Cell key={s.segment} fill={s.color ?? CHART_COLORS[i % CHART_COLORS.length]} stroke="transparent" />
            ))}
          </Pie>
          <Tooltip content={({ active, payload }) => {
            if (!active || !payload?.length) return null;
            const d = payload[0].payload;
            return (
              <div style={{ background:'#1a1a2e', border:'1px solid #2a2a45', borderRadius:10, padding:'10px 14px', fontSize:'.78rem' }}>
                <div style={{ color: d.color, fontWeight:700 }}>{d.segment}</div>
                <div style={{ color:'#e2e8f0' }}>{d.count} customers ({d.pct}%)</div>
                <div style={{ color:'#94a3b8' }}>Revenue: ${d.total_revenue?.toLocaleString()}</div>
              </div>
            );
          }} />
          <Legend formatter={(value) => <span style={{ color:'#94a3b8', fontSize:'.75rem' }}>{value}</span>} />
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
}

// ─── Avg Monetary by Segment Bar ─────────────────────────────────────────────
export function SegmentRevenueBar({ segments }) {
  const data = [...segments].sort((a, b) => b.avg_monetary - a.avg_monetary);
  return (
    <div className="chart-card">
      <div className="chart-card-title">Avg Spend per Customer by Segment</div>
      <ResponsiveContainer width="100%" height={260}>
        <BarChart data={data} margin={{ left: 10, right: 10, bottom: 30 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e1e35" vertical={false} />
          <XAxis dataKey="segment" tick={{ fill:'#64748b', fontSize:10 }} angle={-25} textAnchor="end" interval={0} />
          <YAxis tick={{ fill:'#64748b', fontSize:10 }} tickFormatter={v => `$${v}`} />
          <Tooltip content={<CustomTooltip prefix="$" />} />
          <Bar dataKey="avg_monetary" name="Avg Spend" radius={[6,6,0,0]}>
            {data.map((s, i) => <Cell key={i} fill={s.color ?? CHART_COLORS[i % CHART_COLORS.length]} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

// ─── RFM Scatter ─────────────────────────────────────────────────────────────
export function RFMScatter({ rfmData }) {
  // Sample for perf
  const sample = rfmData.length > 500 ? rfmData.filter((_, i) => i % Math.ceil(rfmData.length / 500) === 0) : rfmData;
  const bySegment = {};
  for (const r of sample) {
    if (!bySegment[r.segment]) bySegment[r.segment] = [];
    bySegment[r.segment].push({ x: r.recency, y: r.monetary, z: r.frequency, name: r.segment });
  }

  return (
    <div className="chart-card">
      <div className="chart-card-title">Recency vs. Monetary (RFM Scatter)</div>
      <ResponsiveContainer width="100%" height={260}>
        <ScatterChart margin={{ left: 10, right: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e1e35" />
          <XAxis dataKey="x" name="Recency (days)" tick={{ fill:'#64748b', fontSize:10 }} label={{ value:'Recency (days)', position:'insideBottom', offset:-5, fill:'#64748b', fontSize:10 }} />
          <YAxis dataKey="y" name="Monetary ($)" tick={{ fill:'#64748b', fontSize:10 }} tickFormatter={v => `$${v}`} />
          <ZAxis dataKey="z" range={[20, 120]} />
          <Tooltip cursor={{ strokeDasharray:'3 3' }} content={({ active, payload }) => {
            if (!active || !payload?.length) return null;
            const d = payload[0].payload;
            return (
              <div style={{ background:'#1a1a2e', border:'1px solid #2a2a45', borderRadius:10, padding:'10px 14px', fontSize:'.78rem' }}>
                <div style={{ color: SEGMENT_META[d.name]?.color ?? '#6366f1', fontWeight:700 }}>{d.name}</div>
                <div style={{ color:'#e2e8f0' }}>Recency: {d.x}d | Spend: ${d.y?.toFixed(0)}</div>
                <div style={{ color:'#94a3b8' }}>Frequency: {d.z}</div>
              </div>
            );
          }} />
          {Object.entries(bySegment).map(([seg, pts]) => (
            <Scatter key={seg} name={seg} data={pts}
              fill={SEGMENT_META[seg]?.color ?? '#6366f1'} opacity={0.6} />
          ))}
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}

// ─── Monthly Revenue Trend ────────────────────────────────────────────────────
export function RevenueTrendChart({ data }) {
  return (
    <div className="chart-card">
      <div className="chart-card-title">Monthly Revenue Trend</div>
      <ResponsiveContainer width="100%" height={220}>
        <AreaChart data={data} margin={{ left: 10, right: 20 }}>
          <defs>
            <linearGradient id="revGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#6366f1" stopOpacity={0.3} />
              <stop offset="100%" stopColor="#6366f1" stopOpacity={0.02} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e1e35" vertical={false} />
          <XAxis dataKey="month" tick={{ fill:'#64748b', fontSize:10 }} />
          <YAxis tick={{ fill:'#64748b', fontSize:10 }} tickFormatter={v => `$${(v/1000).toFixed(0)}K`} />
          <Tooltip content={<CustomTooltip prefix="$" />} />
          <Area type="monotone" dataKey="revenue" name="Revenue" stroke="#6366f1" strokeWidth={2} fill="url(#revGrad)" />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

// ─── Category Revenue Bar ─────────────────────────────────────────────────────
export function CategoryRevenueBar({ data }) {
  const chartData = data.map(([cat, rev]) => ({ cat, revenue: +rev.toFixed(2) }));
  return (
    <div className="chart-card">
      <div className="chart-card-title">Revenue by Category</div>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={chartData} layout="vertical" margin={{ left: 60, right: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e1e35" horizontal={false} />
          <XAxis type="number" tick={{ fill:'#64748b', fontSize:10 }} tickFormatter={v => `$${(v/1000).toFixed(0)}K`} />
          <YAxis type="category" dataKey="cat" tick={{ fill:'#94a3b8', fontSize:11 }} width={56} />
          <Tooltip content={<CustomTooltip prefix="$" />} />
          <Bar dataKey="revenue" name="Revenue" radius={[0,6,6,0]}>
            {chartData.map((_, i) => <Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

// ─── Radar: Segment Comparison ────────────────────────────────────────────────
export function SegmentRadar({ segments }) {
  // Normalize each dimension to 0-100
  const dims = ['avg_recency', 'avg_frequency', 'avg_monetary', 'avg_clv', 'avg_loyalty'];
  const labels = { avg_recency: 'Recency*', avg_frequency: 'Frequency', avg_monetary: 'Spend', avg_clv: 'CLV', avg_loyalty: 'Loyalty' };
  const maxes = {};
  for (const d of dims) maxes[d] = Math.max(...segments.map(s => s[d] ?? 0)) || 1;
  // For recency lower = better, so invert
  const radarData = dims.map(d => {
    const entry = { metric: labels[d] };
    for (const s of segments.slice(0, 4)) {
      const norm = ((s[d] ?? 0) / maxes[d]) * 100;
      entry[s.segment] = d === 'avg_recency' ? 100 - norm : norm;
    }
    return entry;
  });

  return (
    <div className="chart-card">
      <div className="chart-card-title">Segment Comparison Radar</div>
      <ResponsiveContainer width="100%" height={260}>
        <RadarChart data={radarData}>
          <PolarGrid stroke="#1e1e35" />
          <PolarAngleAxis dataKey="metric" tick={{ fill:'#64748b', fontSize:10 }} />
          <PolarRadiusAxis angle={30} domain={[0,100]} tick={{ fill:'#2a2a45', fontSize:9 }} />
          {segments.slice(0, 4).map((s, i) => (
            <Radar key={s.segment} name={s.segment} dataKey={s.segment}
              stroke={s.color ?? CHART_COLORS[i]} fill={s.color ?? CHART_COLORS[i]} fillOpacity={0.12} strokeWidth={2} />
          ))}
          <Legend formatter={v => <span style={{ color:'#94a3b8', fontSize:'.73rem' }}>{v}</span>} />
          <Tooltip content={<CustomTooltip suffix="%" />} />
        </RadarChart>
      </ResponsiveContainer>
      <div style={{ fontSize:'.68rem', color:'var(--text3)', marginTop:'.5rem' }}>* Recency inverted: higher = more recent</div>
    </div>
  );
}

// ─── Channel Bar ──────────────────────────────────────────────────────────────
export function ChannelBar({ data }) {
  const chartData = data.map(([ch, rev]) => ({ channel: ch, revenue: +rev.toFixed(2) }));
  return (
    <div className="chart-card">
      <div className="chart-card-title">Revenue by Channel</div>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={chartData} margin={{ left: 10, right: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e1e35" vertical={false} />
          <XAxis dataKey="channel" tick={{ fill:'#94a3b8', fontSize:11 }} />
          <YAxis tick={{ fill:'#64748b', fontSize:10 }} tickFormatter={v => `$${(v/1000).toFixed(0)}K`} />
          <Tooltip content={<CustomTooltip prefix="$" />} />
          <Bar dataKey="revenue" name="Revenue" radius={[6,6,0,0]}>
            {chartData.map((_, i) => <Cell key={i} fill={['#22d3ee','#6366f1','#f59e0b'][i % 3]} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

// ─── CLV Distribution ─────────────────────────────────────────────────────────
export function CLVHistogram({ rfmData }) {
  const clvs = rfmData.map(r => r.clv).filter(v => v < 1_000_000);
  const min = Math.min(...clvs), max = Math.max(...clvs);
  const bins = 20;
  const step = (max - min) / bins;
  const buckets = Array.from({ length: bins }, (_, i) => ({
    range: `$${Math.round(min + i * step).toLocaleString()}`,
    count: clvs.filter(v => v >= min + i * step && v < min + (i + 1) * step).length,
  }));

  return (
    <div className="chart-card">
      <div className="chart-card-title">Customer Lifetime Value Distribution</div>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={buckets} margin={{ left: 10, right: 10, bottom: 30 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e1e35" vertical={false} />
          <XAxis dataKey="range" tick={{ fill:'#64748b', fontSize:9 }} angle={-35} textAnchor="end" interval={3} />
          <YAxis tick={{ fill:'#64748b', fontSize:10 }} />
          <Tooltip content={<CustomTooltip />} />
          <Bar dataKey="count" name="Customers" fill="#a855f7" radius={[3,3,0,0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
