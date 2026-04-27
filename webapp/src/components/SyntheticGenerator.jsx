import { useState } from 'react';
import { Wand2, Users, ShoppingCart } from 'lucide-react';
import { generateSyntheticData } from '../utils/syntheticData';

export default function SyntheticGenerator({ onDataLoaded }) {
  const [nCustomers, setNCustomers]     = useState(500);
  const [nTransactions, setNTransactions] = useState(2500);
  const [loading, setLoading]           = useState(false);

  const handleGenerate = () => {
    setLoading(true);
    // Defer to next tick so UI updates first
    setTimeout(() => {
      const { transactions } = generateSyntheticData(nCustomers, nTransactions);
      onDataLoaded(transactions, `Synthetic (${nCustomers} customers, ${nTransactions} txns)`);
      setLoading(false);
    }, 10);
  };

  return (
    <div className="synth-card">
      <div className="section-header" style={{ marginBottom: '1.25rem' }}>
        <div>
          <div className="section-title"><span>Generate</span> Synthetic Data</div>
          <div className="section-sub">Realistic retail transactions for demo</div>
        </div>
        <Wand2 size={18} color="var(--text3)" />
      </div>

      <div className="synth-controls">
        <div className="slider-row">
          <div className="slider-label">
            <span style={{ display: 'flex', alignItems: 'center', gap: '.4rem' }}>
              <Users size={13} /> Customers
            </span>
            <span className="slider-val">{nCustomers.toLocaleString()}</span>
          </div>
          <input type="range" min={100} max={2000} step={50}
            value={nCustomers} onChange={e => setNCustomers(+e.target.value)} />
        </div>

        <div className="slider-row">
          <div className="slider-label">
            <span style={{ display: 'flex', alignItems: 'center', gap: '.4rem' }}>
              <ShoppingCart size={13} /> Transactions
            </span>
            <span className="slider-val">{nTransactions.toLocaleString()}</span>
          </div>
          <input type="range" min={500} max={10000} step={250}
            value={nTransactions} onChange={e => setNTransactions(+e.target.value)} />
        </div>

        <div style={{ display: 'flex', gap: '.5rem', marginTop: '.25rem', fontSize: '.73rem', color: 'var(--text3)' }}>
          <span>6 categories</span>
          <span>·</span>
          <span>3 channels</span>
          <span>·</span>
          <span>7 cities</span>
          <span>·</span>
          <span>2 years data</span>
        </div>

        <button className="btn btn-primary" onClick={handleGenerate} disabled={loading}>
          {loading
            ? <><div className="spinner" style={{ width: 14, height: 14 }} /> Generating...</>
            : <><Wand2 size={14} /> Generate Dataset</>
          }
        </button>
      </div>
    </div>
  );
}
