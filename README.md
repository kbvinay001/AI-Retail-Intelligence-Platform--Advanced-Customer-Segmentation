# RetailIQ — AI Retail Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=flat&logo=react&logoColor=black)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-5-646CFF?style=flat&logo=vite&logoColor=white)](https://vitejs.dev/)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Gemini](https://img.shields.io/badge/Gemini-2.5%20Flash-4285F4?style=flat&logo=google&logoColor=white)](https://aistudio.google.com/)
[![Three.js](https://img.shields.io/badge/Three.js-3D%20Spatial-black?style=flat&logo=three.js&logoColor=white)](https://threejs.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> **Version 3.0 — Enterprise Platform + Interactive React Web App**
> Full-stack AI retail analytics: Python backend engine, FastAPI microservice, and a
> premium dark-matte React dashboard with real-time charts, 3D spatial views, and Gemini AI insights.

---

## ✨ What's Inside

### 🖥️ React Web App (`webapp/`)
A production-grade frontend built with Vite + React, featuring:

| Feature | Tech | Details |
|---------|------|---------|
| **CSV Upload + Synthetic Data** | PapaParse + custom RNG | Drag & drop CSV or generate up to 10K transactions |
| **RFM Scoring Engine (JS)** | Vanilla JS | Full Recency · Frequency · Monetary quintile scoring |
| **8-Segment Classification** | Rule-based ML port | VIP Champions · At Risk · Hibernating · and 5 more |
| **Gemini 2.5 Flash AI Insights** | `@google/generative-ai` | Per-segment strategic recommendations |
| **Framer Motion Animations** | `framer-motion` | Staggered entrance · spring hover · AnimatePresence tabs |
| **Regional Sales Velocity Heatmap** | `@nivo/heatmap` | 5 regions × 7 days · dark-matte purple palette |
| **Real-Time Revenue Stream** | Recharts + `setInterval` | Live data injection every 1.8s · pause/resume |
| **3D Store Network** | `@react-three/fiber` | Pulsing glowing nodes · orbit controls · star field |
| **Dark Matte Aesthetic** | Custom CSS | Inter font · indigo/cyan/purple palette · glassmorphism |

### 🐍 Python Backend Engine (`src/`)
Production-ready analytical modules:

| Module | Purpose |
|--------|---------|
| `retail_intelligence.py` | Core RFM engine, K-Means segmentation, anomaly detection |
| `forecasting_engine.py` | Prophet / ARIMA / XGBoost / Simple MA forecasting |
| `ai_recommendations.py` | GPT-4 recommendations with rule-based fallback |
| `multi_store.py` | Multi-store management, regional roll-ups, benchmarking |
| `security.py` | JWT auth, bcrypt, Fernet field-level encryption, RBAC |
| `api.py` | FastAPI REST microservice with Swagger UI |
| `config.py` | Centralised configuration management |

---

## 🚀 Quick Start

### Option A — Web App (Recommended, No Python needed)

**One-click launch (Windows):**
```
Double-click:  start_retailiq.bat
```
Opens `http://localhost:5173` automatically.

**Or manually:**
```bash
cd webapp
npm install
npm run dev
```

**Gemini AI Setup (one-time):**
```bash
# Copy the example file
cp webapp/.env.example webapp/.env

# Edit webapp/.env and paste your key:
VITE_GEMINI_API_KEY=AIzaSy...yourKeyHere
```
Get a free key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey).
Once set, Gemini 2.5 Flash insights load automatically — no pasting required.

---

### Option B — Python Backend Engine

**1. Install dependencies:**
```bash
pip install -r requirements.txt
```

**2. Configure environment:**
```bash
cp .env.example .env
# Edit .env with your OPENAI_API_KEY, SECRET_KEY, etc.
```

**3. Run the full demo:**
```bash
python run_demo.py
```

**4. Launch the REST API:**
```bash
cd src && uvicorn api:build_app --factory --reload
# Swagger UI → http://localhost:8000/docs
```

---

## 📁 Project Structure

```
AI RETAIL INTELLIGENCE/
│
├── webapp/                        # React + Vite Web App
│   ├── src/
│   │   ├── App.jsx                # Main app shell + tab navigation
│   │   ├── index.css              # Dark-matte design system
│   │   ├── motion/
│   │   │   └── variants.js        # Framer-motion animation variants
│   │   ├── components/
│   │   │   ├── KPICards.jsx       # Animated metric cards
│   │   │   ├── Charts.jsx         # Recharts visualizations
│   │   │   ├── NivoCharts.jsx     # Nivo heatmap + real-time line
│   │   │   ├── StoreNetwork3D.jsx # react-three-fiber 3D scene
│   │   │   ├── SegmentCard.jsx    # Segment card + Gemini insight
│   │   │   ├── DataUploader.jsx   # CSV drag & drop
│   │   │   └── SyntheticGenerator.jsx
│   │   └── utils/
│   │       ├── rfm.js             # RFM engine (JS port)
│   │       ├── syntheticData.js   # Seeded data generator
│   │       └── gemini.js          # Gemini API client + fallbacks
│   ├── .env.example               # Copy to .env and add your key
│   └── package.json
│
├── src/                           # Python Backend Engine
│   ├── retail_intelligence.py
│   ├── forecasting_engine.py
│   ├── ai_recommendations.py
│   ├── multi_store.py
│   ├── security.py
│   ├── api.py
│   └── config.py
│
├── tests/                         # 39 pytest test suites
├── docs/                          # API & deployment documentation
├── data/                          # Sample datasets
│
├── run_demo.py                    # Full Python pipeline demo
├── start_retailiq.bat             # One-click Windows launcher
├── requirements.txt
├── setup.py
└── .env.example
```

---

## 🎨 Web App Screens

### Overview Dashboard
- **6 KPI cards** — staggered entrance with spring hover (framer-motion)
- **Real-time revenue chart** — live data injection every 1.8s with pause/resume
- **Regional sales velocity heatmap** — @nivo · 5 regions × 7 days · purple scale
- **Segment distribution** · **RFM scatter** · **Monthly trend** · **Category/Channel bars** · **CLV histogram**

### 3D Spatial Tab
- Interactive **supply chain network** rendered with WebGL (react-three-fiber)
- 6 city store nodes with **pulsing glow animations** and orbit rings
- Auto-rotates when idle · click to inspect revenue · drag to orbit · scroll to zoom
- Wrapped in **React Suspense** — never blocks initial page render

### Segments Tab
- 8 segment cards with **spring hover micro-interactions**
- Per-segment stats: recency · frequency · spend · CLV · loyalty score
- **Gemini 2.5 Flash AI insight** per card: summary · opportunity · 3 action items · priority

### RFM Table Tab
- Full paginated customer table with R/F/M scores (1–5)
- Colour-coded segment badges

---

## 🧪 Python Tests

```bash
python -m pytest tests/ -v
# ✅ 39/39 tests pass
```

| Test Suite | Coverage |
|-----------|---------|
| `test_rfm.py` | RFM computation, CLV, recency |
| `test_segmentation.py` | K-Means, anomaly detection |
| `test_forecasting.py` | All 5 forecast scenarios |
| `test_security.py` | JWT, encryption, RBAC, bcrypt |
| `test_multi_store.py` | Network KPIs, benchmarking |
| `test_recommendations.py` | Rule engine, priority logic |
| `test_api.py` | Full FastAPI integration tests |

---

## 🔐 Security

| Feature | Implementation |
|---------|--------------|
| JWT Authentication | `python-jose` · HS256 · 30-min access tokens |
| Password Hashing | `passlib` · bcrypt/sha256_crypt with auto-fallback |
| Field-level Encryption | `cryptography.fernet` · symmetric AES-128 |
| Role-Based Access Control | Admin · Manager · Analyst · Viewer |
| API Rate Limiting | `slowapi` · per-IP limits |
| Secret Management | `.env` files · never committed to git |

---

## 📊 Python Demo Output

```
==============================================================
  AI RETAIL INTELLIGENCE PLATFORM  -  VERSION 3.0
==============================================================

  DEMO 1: Single-Store Full Pipeline
[OK] Platform initialized — Store: FLAGSHIP_001
[DATA] Generated 1000 customers, 5000 transactions
[RFM]  Computed for 994 customers
[SEG]  KMeans: optimal k=4, silhouette=0.307
[ANOMALY] Detected 100 anomalous customers (10% rate)

  BUSINESS OVERVIEW
   Total Customers  : 994
   Total Revenue    : $375,252.82
   Avg Order Value  : $75.05
   Total CLV        : $30,352,309.82

  DEMO 2: 90-Day Revenue Forecast
   Projected revenue : $48,124.12
   Growth trend      : +9.8%

  DEMO 4: Multi-Store Network (5 stores)
   #1 Delhi Central   : Score 100.0/100
   South Region       : $344,224 across 3 stores

  39/39 tests pass ✅
```

---

## 🛣️ Roadmap

- [ ] PostgreSQL / Supabase integration (replace in-memory store)
- [ ] Export dashboard as PDF report
- [ ] Real-time WebSocket data feed from POS systems
- [ ] Docker compose for one-command full-stack deploy
- [ ] Multi-tenant SaaS mode with per-org isolation

---

## 📄 License

MIT © 2024 — [kbvinay001](https://github.com/kbvinay001)
