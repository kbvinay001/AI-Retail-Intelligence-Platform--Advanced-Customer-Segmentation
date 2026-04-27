import { motion } from 'framer-motion';
import { TrendingUp, TrendingDown, Users, DollarSign, AlertTriangle, Star, ShoppingBag, Activity } from 'lucide-react';
import { staggerContainer, itemVariants, kpiCardHover } from '../motion/variants';

function fmt(n, prefix = '', suffix = '', compact = false) {
  if (n === undefined || n === null || isNaN(n)) return '—';
  if (compact && n >= 1_000_000) return prefix + (n / 1_000_000).toFixed(1) + 'M' + suffix;
  if (compact && n >= 1_000)     return prefix + (n / 1_000).toFixed(1) + 'K' + suffix;
  return prefix + n.toLocaleString(undefined, { maximumFractionDigits: 2 }) + suffix;
}

const CARDS = [
  { key: 'totalCustomers', label: 'Total Customers',    icon: Users,          color: '#6366f1', format: v => fmt(v),              sub: 'unique buyer profiles',    badge: null },
  { key: 'totalRevenue',   label: 'Total Revenue',      icon: DollarSign,     color: '#22d3ee', format: v => fmt(v,'$','',true),  sub: 'from all transactions',    badge: null },
  { key: 'avgOrderValue',  label: 'Avg Order Value',    icon: ShoppingBag,    color: '#f59e0b', format: v => fmt(v,'$'),          sub: 'per transaction',
    badge: v => ({ text: v > 75 ? 'Healthy AOV' : 'Grow AOV', cls: v > 75 ? 'up' : 'warn' }) },
  { key: 'totalCLV',       label: 'Total CLV',          icon: Activity,       color: '#a855f7', format: v => fmt(v,'$','',true),  sub: 'projected lifetime value', badge: null },
  { key: 'avgCLV',         label: 'Avg CLV / Customer', icon: Star,           color: '#34d399', format: v => fmt(v,'$','',true),  sub: 'per customer',             badge: null },
  { key: 'atRiskCount',    label: 'At-Risk Customers',  icon: AlertTriangle,  color: '#f43f5e', format: v => fmt(v),              sub: '180+ days inactive',
    badge: (v, kpis) => ({ text: `${((v / kpis.totalCustomers) * 100).toFixed(0)}% of base`, cls: v / kpis.totalCustomers > 0.25 ? 'down' : 'warn' }) },
];

export default function KPICards({ kpis }) {
  if (!kpis) return null;

  return (
    <motion.div
      className="kpi-grid"
      variants={staggerContainer}
      initial="hidden"
      animate="show"
    >
      {CARDS.map(({ key, label, icon: Icon, color, format, sub, badge }) => {
        const val = kpis[key];
        const b   = badge ? badge(val, kpis) : null;

        return (
          <motion.div
            className="kpi-card"
            key={key}
            variants={itemVariants}
            whileHover={kpiCardHover}
            style={{ cursor: 'default' }}
          >
            {/* Ambient glow */}
            <motion.div
              className="kpi-card-glow"
              style={{ background: color }}
              animate={{ opacity: [0.08, 0.18, 0.08] }}
              transition={{ duration: 3, repeat: Infinity, ease: 'easeInOut' }}
            />

            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '.5rem' }}>
              <div className="kpi-label">{label}</div>
              <motion.div whileHover={{ rotate: 15, scale: 1.2 }} transition={{ type: 'spring', stiffness: 300 }}>
                <Icon size={16} color={color} opacity={0.7} />
              </motion.div>
            </div>

            <motion.div
              className="kpi-value"
              style={{ color }}
              initial={{ opacity: 0, scale: 0.85 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.1, type: 'spring', stiffness: 200 }}
            >
              {format(val)}
            </motion.div>

            <div className="kpi-sub">{sub}</div>
            {b && <div className={`kpi-badge ${b.cls}`}>{b.text}</div>}
          </motion.div>
        );
      })}
    </motion.div>
  );
}
