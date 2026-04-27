# API Documentation — V3.0

## Base URL
```
http://localhost:8000/api/v3
```

## Authentication
All endpoints (except `/health` and `/auth/login`) require a **Bearer JWT token**.

```bash
# Login
curl -X POST http://localhost:8000/api/v3/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"Admin@123"}'

# Use token
curl -H "Authorization: Bearer <token>" http://localhost:8000/api/v3/auth/me
```

## Demo Credentials
| Username | Password   | Role    |
|----------|-----------|---------|
| admin    | Admin@123 | ADMIN   |
| analyst  | Analyst@123 | ANALYST |

## Endpoints

### System
| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |

### Auth
| Method | Path | Description |
|--------|------|-------------|
| POST | `/auth/login` | Get JWT token |
| GET  | `/auth/me`    | Current user info |

### Data
| Method | Path | Description |
|--------|------|-------------|
| POST | `/data/generate` | Generate synthetic data |

### Analytics
| Method | Path | Description |
|--------|------|-------------|
| POST | `/analytics/{store_id}/rfm` | Compute RFM metrics |
| POST | `/analytics/{store_id}/segment` | Run ML segmentation |
| GET  | `/analytics/{store_id}/insights` | Business KPIs |
| GET  | `/analytics/{store_id}/anomalies` | Anomaly detection |

### Forecasting
| Method | Path | Description |
|--------|------|-------------|
| POST | `/forecast/{store_id}` | Revenue forecast |

### AI
| Method | Path | Description |
|--------|------|-------------|
| GET | `/recommendations/{store_id}` | GPT/Rule-based strategy recs |

## Interactive Docs
Open `http://localhost:8000/docs` for the full Swagger UI after starting the API.
