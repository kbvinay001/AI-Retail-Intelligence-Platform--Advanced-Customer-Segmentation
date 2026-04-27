# Deployment Guide — V3.0

## Quick Start (Local)

```bash
# 1. Clone
git clone https://github.com/kbvinay001/AI-Retail-Intelligence-Platform--Advanced-Customer-Segmentation.git
cd AI-Retail-Intelligence-Platform--Advanced-Customer-Segmentation

# 2. Install core dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env with your values

# 4. Run the demo
python run_demo.py

# 5. Start the REST API
python src/api.py
# → Open http://localhost:8000/docs
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| SECRET_KEY | Yes | — | JWT signing key |
| OPENAI_API_KEY | No | — | Enables GPT-4 recommendations |
| ENCRYPTION_KEY | No | Auto | Fernet key for field encryption |
| DATABASE_URL | No | SQLite | Database connection string |
| REDIS_URL | No | localhost | Cache/task broker |

## Google Colab

```python
!pip install plotly scikit-learn pandas numpy matplotlib seaborn fastapi python-jose passlib cryptography python-dotenv
!git clone https://github.com/kbvinay001/AI-Retail-Intelligence-Platform--Advanced-Customer-Segmentation.git
%cd AI-Retail-Intelligence-Platform--Advanced-Customer-Segmentation
exec(open('run_demo.py').read())
```

## Docker (Optional)

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "src/api.py"]
```

```bash
docker build -t retail-intelligence:v3 .
docker run -p 8000:8000 --env-file .env retail-intelligence:v3
```

## Running Tests

```bash
python -m pytest tests/ -v --cov=src
```
