import { useState, useMemo, lazy, Suspense } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Key, RefreshCw, BarChart2, Layers, Table2, Globe } from 'lucide-react';
import DataUploader from './components/DataUploader';
import SyntheticGenerator from './components/SyntheticGenerator';
import KPICards from './components/KPICards';
import SegmentCard from './components/SegmentCard';
import {
  SegmentPieChart, SegmentRevenueBar, RFMScatter,
  RevenueTrendChart, CategoryRevenueBar, SegmentRadar,
  ChannelBar, CLVHistogram,
} from './components/Charts';
import { SalesVelocityHeatmap, RealTimeRevenueChart } from './components/NivoCharts';
import { computeRFM, aggregateSegments, computeKPIs, monthlyRevenue } from './utils/rfm';
import {
  pageVariants, staggerContainer, itemVariants,
  slideInLeft, slideInRight, tabContent, fadeUp,
} from './motion/variants';

// Lazy-load heavy 3D component
const StoreNetwork3D = lazy(() => import('./components/StoreNetwork3D'));

// ─── Navbar ───────────────────────────────────────────────────────────────────
function Navbar({ apiKey, setApiKey }) {
  const [show, setShow] = useState(false);
  const isActive = apiKey.trim().length > 10;

  return (
    <motion.nav
      className="navbar"
      initial={{ opacity: 0, y: -16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: [0.23, 1, 0.32, 1] }}
    >
      <div className="navbar-brand">
        <motion.div
          className="navbar-logo"
          whileHover={{ rotate: [0, -10, 10, 0], scale: 1.1 }}
          transition={{ duration: 0.5 }}
        >
          🏪
        </motion.div>
        <div>
          <div className="navbar-title">Retail<span>IQ</span></div>
        </div>
        <span className="navbar-badge">V3.0</span>
      </div>

      <div className="navbar-right">
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: '.3rem' }}>
          <div style={{ fontSize: '.67rem', color: 'var(--text3)', display: 'flex', alignItems: 'center', gap: '.4rem' }}>
            <span>Gemini API Key</span>
            <AnimatePresence mode="wait">
              {isActive ? (
                <motion.span key="active"
                  initial={{ opacity: 0, scale: 0.8 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.8 }}
                  style={{ background: 'rgba(52,211,153,.15)', color: '#34d399', border: '1px solid rgba(52,211,153,.3)', borderRadius: 99, padding: '0 .45rem', fontSize: '.62rem', fontWeight: 700 }}>
                  ✓ gemini-2.5-flash active
                </motion.span>
              ) : (
                <motion.span key="inactive"
                  initial={{ opacity: 0, scale: 0.8 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.8 }}
                  style={{ background: 'rgba(245,158,11,.1)', color: '#f59e0b', border: '1px solid rgba(245,158,11,.25)', borderRadius: 99, padding: '0 .45rem', fontSize: '.62rem', fontWeight: 600 }}>
                  Smart rules (no key)
                </motion.span>
              )}
            </AnimatePresence>
          </div>
          <div className="api-input-wrap" style={{ borderColor: isActive ? 'rgba(52,211,153,.4)' : undefined }}>
            <Key size={13} />
            <input
              className="api-input"
              type={show ? 'text' : 'password'}
              placeholder="Paste Gemini API key here…"
              value={apiKey}
              onChange={e => setApiKey(e.target.value)}
              onFocus={() => setShow(true)}
              onBlur={() => setShow(false)}
            />
            {isActive && <motion.span animate={{ opacity: [1, 0.4, 1] }} transition={{ duration: 1.5, repeat: Infinity }} style={{ fontSize: '.65rem', color: '#34d399' }}>●</motion.span>}
          </div>
        </div>
      </div>
    </motion.nav>
  );
}

// ─── Hero ─────────────────────────────────────────────────────────────────────
function Hero({ onDemoClick }) {
  return (
    <motion.div
      className="hero"
      variants={staggerContainer}
      initial="hidden"
      animate="show"
    >
      <motion.div className="hero-tag" variants={itemVariants}>
        ✦ AI-Powered · RFM Segmentation · Gemini Insights · 3D Spatial
      </motion.div>
      <motion.h1 variants={itemVariants}>
        Turn Customer Data Into<br />
        <span className="grad">Actionable Intelligence</span>
      </motion.h1>
      <motion.p variants={itemVariants}>
        Upload your transaction CSV or generate synthetic data. Get instant RFM scoring,
        ML-style segmentation, AI-powered strategic recommendations, and immersive 3D spatial views.
      </motion.p>
      <motion.div className="hero-actions" variants={itemVariants}>
        <motion.button
          className="btn btn-primary"
          onClick={onDemoClick}
          whileHover={{ scale: 1.04, boxShadow: '0 8px 32px rgba(99,102,241,0.45)' }}
          whileTap={{ scale: 0.97 }}
        >
          ⚡ Quick Demo (Synthetic Data)
        </motion.button>
      </motion.div>
    </motion.div>
  );
}

// ─── 3D Suspense Spinner ──────────────────────────────────────────────────────
function CanvasLoader() {
  return (
    <div style={{ height: 460, display: 'flex', alignItems: 'center', justifyContent: 'center',
      background: 'var(--card)', borderRadius: 'var(--radius-lg)', border: '1px solid var(--border)',
      flexDirection: 'column', gap: '1rem' }}>
      <motion.div
        style={{ width: 48, height: 48, borderRadius: '50%', border: '2px solid var(--border2)', borderTopColor: 'var(--accent)' }}
        animate={{ rotate: 360 }}
        transition={{ duration: 0.9, repeat: Infinity, ease: 'linear' }}
      />
      <div style={{ fontSize: '.78rem', color: 'var(--text3)' }}>Loading 3D spatial engine…</div>
    </div>
  );
}

// ─── RFM Table ────────────────────────────────────────────────────────────────
function RFMTable({ rfmData }) {
  const [page, setPage] = useState(0);
  const PER = 12;
  const total = rfmData.length;
  const slice = rfmData.slice(page * PER, (page + 1) * PER);
  return (
    <motion.div className="chart-card" style={{ marginBottom: '1.5rem' }}
      variants={fadeUp} initial="hidden" animate="show">
      <div className="section-header">
        <div className="chart-card-title" style={{ marginBottom: 0 }}>Customer RFM Table</div>
        <div style={{ fontSize: '.75rem', color: 'var(--text3)' }}>
          {total.toLocaleString()} customers · page {page + 1}/{Math.ceil(total / PER)}
        </div>
      </div>
      <div className="data-table-wrap">
        <table className="data-table">
          <thead>
            <tr>{['Customer ID','Segment','Recency','Frequency','Monetary','Loyalty','CLV','R','F','M'].map(h => <th key={h}>{h}</th>)}</tr>
          </thead>
          <tbody>
            {slice.map(r => (
              <tr key={r.customer_id}>
                <td className="mono" style={{ color: 'var(--text)' }}>#{r.customer_id}</td>
                <td>
                  <span className="badge" style={{ background: `${r.color ?? '#6366f1'}18`, color: r.color ?? '#6366f1', border: `1px solid ${r.color ?? '#6366f1'}30` }}>
                    {r.segment}
                  </span>
                </td>
                <td>{r.recency}d</td>
                <td>{r.frequency}</td>
                <td>${r.monetary.toFixed(0)}</td>
                <td>{r.loyalty_score}</td>
                <td>${Math.round(r.clv).toLocaleString()}</td>
                <td className="mono" style={{ color: '#6366f1' }}>{r.r_score}</td>
                <td className="mono" style={{ color: '#22d3ee' }}>{r.f_score}</td>
                <td className="mono" style={{ color: '#a855f7' }}>{r.m_score}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div style={{ display: 'flex', gap: '.5rem', marginTop: '1rem', justifyContent: 'flex-end' }}>
        <button className="btn btn-outline btn-sm" disabled={page === 0} onClick={() => setPage(p => p - 1)}>← Prev</button>
        <button className="btn btn-outline btn-sm" disabled={(page + 1) * PER >= total} onClick={() => setPage(p => p + 1)}>Next →</button>
      </div>
    </motion.div>
  );
}

// ─── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  const [transactions, setTransactions] = useState(null);
  const [dataLabel, setDataLabel]       = useState('');
  const [apiKey, setApiKey]             = useState(
    () => import.meta.env.VITE_GEMINI_API_KEY || ''
  );
  const [activeTab, setActiveTab]       = useState('overview');

  const analytics = useMemo(() => {
    if (!transactions) return null;
    const rfmData  = computeRFM(transactions);
    const segs     = aggregateSegments(rfmData);
    const kpis     = computeKPIs(rfmData, transactions);
    const monthly  = monthlyRevenue(transactions);
    const segColor = {};
    for (const s of segs) segColor[s.segment] = s.color;
    const rfmWithColor = rfmData.map(r => ({ ...r, color: segColor[r.segment] }));
    return { rfmData: rfmWithColor, segments: segs, kpis, monthly };
  }, [transactions]);

  const handleDataLoaded = (txns, label) => {
    setTransactions(txns);
    setDataLabel(label);
    setActiveTab('overview');
  };

  const handleReset = () => { setTransactions(null); setDataLabel(''); };

  const handleDemoClick = () => {
    import('./utils/syntheticData').then(m => {
      const { transactions } = m.generateSyntheticData(500, 2500);
      handleDataLoaded(transactions, 'Synthetic Demo (500 customers, 2500 txns)');
    });
  };

  const TABS = [
    { id: 'overview', label: 'Overview',   icon: BarChart2 },
    { id: 'spatial',  label: '3D Spatial', icon: Globe },
    { id: 'segments', label: 'Segments',   icon: Layers },
    { id: 'table',    label: 'RFM Table',  icon: Table2 },
  ];

  return (
    <div className="app-shell">
      <Navbar apiKey={apiKey} setApiKey={setApiKey} />

      {/* ── Landing ── */}
      <AnimatePresence mode="wait">
        {!transactions && (
          <motion.div key="landing" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0, y: -20 }}>
            <Hero onDemoClick={handleDemoClick} />
            <motion.div
              className="upload-section"
              variants={staggerContainer}
              initial="hidden"
              animate="show"
            >
              <motion.div variants={slideInLeft}><DataUploader onDataLoaded={handleDataLoaded} /></motion.div>
              <motion.div variants={slideInRight}><SyntheticGenerator onDataLoaded={handleDataLoaded} /></motion.div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ── Dashboard ── */}
      <AnimatePresence mode="wait">
        {transactions && analytics && (
          <motion.div
            key="dashboard"
            variants={pageVariants}
            initial="hidden"
            animate="show"
            exit={{ opacity: 0, y: -10 }}
          >
            {/* Header */}
            <motion.div
              style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.75rem', flexWrap: 'wrap', gap: '1rem' }}
              variants={itemVariants}
            >
              <div>
                <h2 style={{ fontSize: '1.3rem', fontWeight: 700, letterSpacing: '-.02em' }}>Analytics Dashboard</h2>
                <div style={{ fontSize: '.78rem', color: 'var(--text3)', marginTop: '.2rem' }}>
                  {dataLabel} · {analytics.kpis.totalCustomers.toLocaleString()} customers · {transactions.length.toLocaleString()} transactions
                </div>
              </div>
              <motion.button
                className="btn btn-outline btn-sm"
                onClick={handleReset}
                whileHover={{ scale: 1.03 }}
                whileTap={{ scale: 0.97 }}
              >
                <RefreshCw size={13} /> New Analysis
              </motion.button>
            </motion.div>

            {/* Tabs */}
            <motion.div className="tab-bar" variants={itemVariants}>
              {TABS.map(({ id, label, icon: Icon }) => (
                <motion.button
                  key={id}
                  className={`tab ${activeTab === id ? 'active' : ''}`}
                  onClick={() => setActiveTab(id)}
                  whileHover={{ scale: 1.04 }}
                  whileTap={{ scale: 0.97 }}
                >
                  <Icon size={13} style={{ display: 'inline', marginRight: 4 }} />
                  {label}
                </motion.button>
              ))}
            </motion.div>

            {/* Tab content with AnimatePresence */}
            <AnimatePresence mode="wait">

              {/* ── Overview ── */}
              {activeTab === 'overview' && (
                <motion.div key="overview" variants={tabContent} initial="hidden" animate="show" exit="exit">
                  <KPICards kpis={analytics.kpis} />

                  {/* Agent 2: Nivo charts row */}
                  <div className="charts-grid" style={{ marginBottom: '1.25rem' }}>
                    <RealTimeRevenueChart />
                    <SalesVelocityHeatmap />
                  </div>

                  <div className="charts-grid">
                    <SegmentPieChart segments={analytics.segments} />
                    <SegmentRevenueBar segments={analytics.segments} />
                  </div>
                  <div className="charts-grid">
                    <RFMScatter rfmData={analytics.rfmData} />
                    <RevenueTrendChart data={analytics.monthly} />
                  </div>
                  <div className="charts-grid-3">
                    <CategoryRevenueBar data={analytics.kpis.categoryRevenue} />
                    <ChannelBar data={analytics.kpis.channelRevenue} />
                    <SegmentRadar segments={analytics.segments} />
                  </div>
                  <CLVHistogram rfmData={analytics.rfmData} />
                </motion.div>
              )}

              {/* ── 3D Spatial ── */}
              {activeTab === 'spatial' && (
                <motion.div key="spatial" variants={tabContent} initial="hidden" animate="show" exit="exit">
                  <div style={{ marginBottom: '1.25rem', fontSize: '.83rem', color: 'var(--text3)' }}>
                    Interactive 3D supply chain · click nodes to inspect · drag to orbit
                  </div>
                  <Suspense fallback={<CanvasLoader />}>
                    <StoreNetwork3D />
                  </Suspense>
                  <div style={{ marginTop: '1.25rem' }} className="charts-grid">
                    <SalesVelocityHeatmap />
                    <RealTimeRevenueChart />
                  </div>
                </motion.div>
              )}

              {/* ── Segments ── */}
              {activeTab === 'segments' && (
                <motion.div key="segments" variants={tabContent} initial="hidden" animate="show" exit="exit">
                  <div style={{ marginBottom: '1.25rem', fontSize: '.83rem', color: 'var(--text3)' }}>
                    {analytics.segments.length} segments ·{' '}
                    {apiKey ? ' Gemini 2.5 Flash insights enabled' : ' Smart rule-based insights (add Gemini key in navbar for live AI)'}
                  </div>
                  <motion.div
                    className="segments-grid"
                    variants={staggerContainer}
                    initial="hidden"
                    animate="show"
                  >
                    {analytics.segments.map(seg => (
                      <motion.div key={seg.segment} variants={itemVariants}>
                        <SegmentCard seg={seg} kpis={analytics.kpis} geminiKey={apiKey} />
                      </motion.div>
                    ))}
                  </motion.div>
                </motion.div>
              )}

              {/* ── RFM Table ── */}
              {activeTab === 'table' && (
                <motion.div key="table" variants={tabContent} initial="hidden" animate="show" exit="exit">
                  <RFMTable rfmData={analytics.rfmData} />
                </motion.div>
              )}

            </AnimatePresence>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
