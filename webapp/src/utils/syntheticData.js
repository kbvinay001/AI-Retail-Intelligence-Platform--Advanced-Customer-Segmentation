/**
 * Synthetic Retail Data Generator
 * Generates realistic customer transaction data for demo purposes.
 */

const CATEGORIES = ['Electronics', 'Clothing', 'Home & Garden', 'Groceries', 'Sports', 'Beauty'];
const CHANNELS   = ['Online', 'In-Store', 'Mobile App'];
const CITIES     = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Hyderabad', 'Pune', 'Kolkata'];
const AGE_GROUPS = ['18-25', '26-35', '36-45', '46-55', '55+'];
const GENDERS    = ['Male', 'Female'];

/** Seeded pseudo-random (simple LCG) so data is reproducible */
class RNG {
  constructor(seed = 42) { this.s = seed; }
  next() { this.s = (this.s * 1664525 + 1013904223) & 0xffffffff; return (this.s >>> 0) / 0xffffffff; }
  int(a, b) { return Math.floor(this.next() * (b - a + 1)) + a; }
  choice(arr) { return arr[this.int(0, arr.length - 1)]; }
  logNormal(mu = 4, sigma = 0.8) {
    // Box-Muller
    const u1 = Math.max(1e-9, this.next()), u2 = this.next();
    const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    return Math.exp(mu + sigma * z);
  }
}

function randomDate(rng, daysBack = 730) {
  const d = new Date();
  d.setDate(d.getDate() - rng.int(1, daysBack));
  return d.toISOString().split('T')[0];
}

export function generateSyntheticData(nCustomers = 500, nTransactions = 2500, seed = 42) {
  const rng = new RNG(seed);

  const customers = Array.from({ length: nCustomers }, (_, i) => ({
    customer_id:      i + 1,
    age_group:        rng.choice(AGE_GROUPS),
    gender:           rng.choice(GENDERS),
    city:             rng.choice(CITIES),
    acquisition_date: randomDate(rng, 1095),
  }));

  const transactions = Array.from({ length: nTransactions }, (_, i) => ({
    transaction_id:   i + 1,
    customer_id:      rng.int(1, nCustomers),
    transaction_date: randomDate(rng, 730),
    amount:           Math.round(rng.logNormal(4.0, 0.8) * 100) / 100,
    quantity:         rng.int(1, 5),
    category:         rng.choice(CATEGORIES),
    channel:          rng.choice(CHANNELS),
  }));

  return { customers, transactions };
}

/** Parse CSV rows into transaction-like objects */
export function parseCSVToTransactions(rows) {
  if (!rows || rows.length === 0) return [];

  const headers = Object.keys(rows[0]).map(h => h.toLowerCase().trim());
  const get = (row, candidates) => {
    for (const c of candidates) {
      const key = Object.keys(row).find(k => k.toLowerCase().trim() === c);
      if (key && row[key] !== undefined && row[key] !== '') return row[key];
    }
    return null;
  };

  return rows
    .map((row, i) => ({
      transaction_id:   i + 1,
      customer_id:      get(row, ['customer_id', 'customerid', 'cust_id', 'customer']) ?? i + 1,
      transaction_date: get(row, ['transaction_date', 'date', 'order_date', 'purchase_date']) ?? new Date().toISOString().split('T')[0],
      amount:           parseFloat(get(row, ['amount', 'total', 'revenue', 'price', 'value', 'sales']) ?? 0),
      quantity:         parseInt(get(row, ['quantity', 'qty', 'units']) ?? 1),
      category:         get(row, ['category', 'product_category', 'dept']) ?? 'General',
      channel:          get(row, ['channel', 'source', 'medium']) ?? 'Unknown',
    }))
    .filter(t => !isNaN(t.amount) && t.amount > 0);
}
