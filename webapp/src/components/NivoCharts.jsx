/**
 * Agent 2: Advanced Data Visualizations
 * - @nivo/heatmap: Regional Sales Velocity (stores × day of week)
 * - Real-time revenue line chart (recharts + live data injection)
 * Both tuned to dark-matte aesthetic with premium styled tooltips.
 */
import { useState, useEffect, useRef, useMemo } from 'react';
import { motion } from 'framer-motion';
import { ResponsiveHeatMap } from '@nivo/heatmap';
import {
  ResponsiveContainer, ComposedChart, Line, Area, XAxis, YAxis,
  CartesianGrid, Tooltip, ReferenceLine,
} from 'recharts';
import { chartCardHover, scaleIn } from '../motion/variants';

// ─── Shared Nivo dark-matte theme ────────────────────────────────────────────
export const nivoTheme = {
  background: 'transparent',
  textColor: '#64748b',
  fontSize: 11,
  fontFamily: 'Inter, sans-serif',
  axis: {
    ticks: { text: { fill: '#64748b', fontSize: 10 } },
    legend: { text: { fill: '#94a3b8', fontSize: 11, fontWeight: 600 } },
  },
  grid:    { line: { stroke: '#1e1e35', strokeWidth: 1 } },
  legends: { text: { fill: '#94a3b8', fontSize: 10 } },
  tooltip: {
    container: {
      background: '#1a1a2e',
      border: '1px solid #2a2a45',
      borderRadius: 10,
      color: '#e2e8f0',
      fontSize: 12,
      boxShadow: '0 8px 32px rgba(0,0,0,0.6)',
      padding: '8px 14px',
    },
  },
  crosshair: { line: { stroke: '#6366f1', strokeWidth: 1, strokeOpacity: 0.5 } },
};

// ─── Heatmap: Regional Sales Velocity ────────────────────────────────────────
const DAYS    = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
const REGIONS = ['Online', 'West', 'North', 'East', 'South'];

function buildHeatmapData(seed = 1) {
  const rng = (x) => (Math.sin(x * seed * 9301 + 49297) * 0.5 + 0.5);
  return REGIONS.map((region, ri) => ({
    id: region,
    data: DAYS.map((day, di) => ({
      x: day,
      y: Math.round(30 + rng(ri * 7 + di) * 70),
    })),
  }));
}

const HeatmapTooltip = ({ cell }) => (
  <div style={{
    background: '#13131f', border: '1px solid #2a2a45',
    borderRadius: 10, padding: '10px 14px', fontSize: 12, color: '#e2e8f0',
    boxShadow: '0 8px 32px rgba(0,0,0,.6)',
    display: 'flex', flexDirection: 'column', gap: 4,
  }}>
    <span style={{ color: '#94a3b8', fontSize: 11 }}>{cell.serieId} · {cell.data.x}</span>
    <span style={{ fontWeight: 700, fontSize: 15, color: '#6366f1' }}>
      {cell.value}<span style={{ fontSize: 11, color: '#64748b', fontWeight: 400 }}> / 100</span>
    </span>
    <span style={{ fontSize: 10, color: '#64748b' }}>Sales Velocity Score</span>
  </div>
);

export function SalesVelocityHeatmap() {
  const data = useMemo(() => buildHeatmapData(42), []);

  return (
    <motion.div
      className="chart-card"
      variants={scaleIn}
      initial="hidden"
      animate="show"
      whileHover={chartCardHover}
    >
      <div className="chart-card-title">
        Regional Sales Velocity Heatmap
        <span style={{ fontSize: '.68rem', color: 'var(--text3)', marginLeft: '.75rem', fontWeight: 400 }}>
          score 0–100 by day of week
        </span>
      </div>

      <div style={{ height: 240 }}>
        <ResponsiveHeatMap
          data={data}
          theme={nivoTheme}
          margin={{ top: 10, right: 20, bottom: 40, left: 70 }}
          axisTop={null}
          axisBottom={{
            tickSize: 0, tickPadding: 8,
            legend: 'Day of Week', legendPosition: 'middle', legendOffset: 34,
          }}
          axisLeft={{
            tickSize: 0, tickPadding: 10,
            legend: 'Region', legendPosition: 'middle', legendOffset: -58,
          }}
          colors={{ type: 'sequential', scheme: 'purples', minValue: 0, maxValue: 100 }}
          emptyColor="#1e1e35"
          borderRadius={4}
          borderWidth={1}
          borderColor="#0f0f1a"
          enableLabels={true}
          labelTextColor={{ from: 'color', modifiers: [['brighter', 3]] }}
          animate={true}
          motionConfig="gentle"
          tooltip={HeatmapTooltip}
          legends={[{
            anchor: 'bottom-right', direction: 'column',
            translateX: 30, translateY: -10,
            length: 120, thickness: 8, ticks: 3,
            tickFormat: v => `${v}`,
          }]}
        />
      </div>
    </motion.div>
  );
}

// ─── Real-time Revenue Line Chart ─────────────────────────────────────────────
const INIT_POINTS = 20;

function generatePoint(i, base = 8000) {
  const trend = i * 50;
  const noise = (Math.sin(i * 0.7) * 800) + (Math.random() - 0.5) * 600;
  return {
    t:       `T+${i}`,
    revenue: Math.round(Math.max(0, base + trend + noise)),
    target:  base + trend + 400,
  };
}

const LiveTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: '#13131f', border: '1px solid #2a2a45', borderRadius: 10,
      padding: '10px 16px', fontSize: 12, color: '#e2e8f0',
      boxShadow: '0 8px 32px rgba(0,0,0,.65)',
    }}>
      <div style={{ color: '#64748b', marginBottom: 6, fontSize: 11 }}>{label}</div>
      {payload.map((p, i) => (
        <div key={i} style={{ color: p.color, fontWeight: 700, fontSize: 13 }}>
          {p.name}: ${p.value?.toLocaleString()}
        </div>
      ))}
    </div>
  );
};

export function RealTimeRevenueChart() {
  const [data, setData]         = useState(() => Array.from({ length: INIT_POINTS }, (_, i) => generatePoint(i)));
  const [isLive, setIsLive]     = useState(true);
  const counterRef              = useRef(INIT_POINTS);
  const lastRevRef              = useRef(data[data.length - 1].revenue);

  useEffect(() => {
    if (!isLive) return;
    const id = setInterval(() => {
      const i    = counterRef.current++;
      const base = lastRevRef.current;
      const next = generatePoint(i, base * 0.98 + 200);
      lastRevRef.current = next.revenue;
      setData(prev => [...prev.slice(-29), next]);
    }, 1800);
    return () => clearInterval(id);
  }, [isLive]);

  const latest = data[data.length - 1];
  const prev   = data[data.length - 2];
  const delta  = prev ? ((latest.revenue - prev.revenue) / prev.revenue * 100).toFixed(1) : 0;
  const up     = delta >= 0;

  return (
    <motion.div
      className="chart-card"
      variants={scaleIn}
      initial="hidden"
      animate="show"
      whileHover={chartCardHover}
    >
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.25rem' }}>
        <div>
          <div className="chart-card-title" style={{ marginBottom: '.2rem' }}>
            Real-Time Revenue Stream
            <motion.span
              animate={{ opacity: isLive ? [1, 0.3, 1] : 1 }}
              transition={{ duration: 1.2, repeat: isLive ? Infinity : 0 }}
              style={{
                display: 'inline-block', width: 7, height: 7, borderRadius: '50%',
                background: isLive ? '#34d399' : '#64748b',
                marginLeft: '.6rem', verticalAlign: 'middle',
              }}
            />
          </div>
          <div style={{ fontSize: '.7rem', color: 'var(--text3)' }}>
            Injecting a new data point every 1.8s
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '.75rem' }}>
          <div style={{ textAlign: 'right' }}>
            <div style={{ fontSize: '.68rem', color: 'var(--text3)' }}>Latest</div>
            <div style={{ fontSize: '1.1rem', fontWeight: 800, color: '#6366f1', fontFamily: 'JetBrains Mono, monospace' }}>
              ${latest.revenue.toLocaleString()}
            </div>
            <div style={{ fontSize: '.68rem', color: up ? '#34d399' : '#f43f5e', fontWeight: 600 }}>
              {up ? '▲' : '▼'} {Math.abs(delta)}%
            </div>
          </div>
          <button
            className="btn btn-outline btn-sm"
            onClick={() => setIsLive(l => !l)}
            style={{ minWidth: 60 }}
          >
            {isLive ? '⏸ Pause' : '▶ Live'}
          </button>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={200}>
        <ComposedChart data={data} margin={{ left: 10, right: 20 }}>
          <defs>
            <linearGradient id="liveGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%"   stopColor="#6366f1" stopOpacity={0.25} />
              <stop offset="100%" stopColor="#6366f1" stopOpacity={0.02} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e1e35" vertical={false} />
          <XAxis dataKey="t" tick={{ fill: '#64748b', fontSize: 9 }} interval={4} />
          <YAxis tick={{ fill: '#64748b', fontSize: 10 }} tickFormatter={v => `$${(v / 1000).toFixed(0)}K`} />
          <Tooltip content={<LiveTooltip />} />
          <ReferenceLine y={latest.target} stroke="#f59e0b" strokeDasharray="4 4" strokeWidth={1.5} />
          <Area  type="monotone" dataKey="revenue" name="Revenue" stroke="#6366f1" strokeWidth={0}  fill="url(#liveGrad)" isAnimationActive={false} />
          <Line  type="monotone" dataKey="revenue" name="Revenue" stroke="#6366f1" strokeWidth={2.5} dot={false} isAnimationActive={false} />
          <Line  type="monotone" dataKey="target"  name="Target"  stroke="#f59e0b" strokeWidth={1.5} dot={false} strokeDasharray="4 4" isAnimationActive={false} />
        </ComposedChart>
      </ResponsiveContainer>
    </motion.div>
  );
}
