"""
V3.0 — Advanced Forecasting Engine
Supports Prophet, ARIMA (pmdarima), and XGBoost-based time-series forecasting.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    import pmdarima as pm
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


class ForecastingEngine:
    """
    Multi-model time-series forecasting for retail KPIs.
    Supports Prophet, ARIMA, and XGBoost.
    """

    def __init__(self, model: str = "simple"):
        """
        model: 'prophet' | 'arima' | 'xgboost' | 'simple'
        Falls back to 'simple' (moving average) if libraries not installed.
        """
        self.model_type = model
        self.fitted_model = None
        self.forecast_df = None

    # ─── Prepare Revenue Time Series ────────────────────────────────────────

    @staticmethod
    def prepare_revenue_series(transactions_df: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
        """Aggregate transactions to daily/weekly/monthly revenue series."""
        ts = transactions_df.copy()
        ts['transaction_date'] = pd.to_datetime(ts['transaction_date'])
        ts = ts.groupby('transaction_date')['amount'].sum().reset_index()
        ts = ts.set_index('transaction_date').resample(freq).sum().fillna(0).reset_index()
        ts.columns = ['ds', 'y']
        return ts

    # ─── Simple Moving-Average Fallback ─────────────────────────────────────

    def _simple_forecast(self, ts: pd.DataFrame, horizon: int = 90) -> pd.DataFrame:
        window = min(30, len(ts))
        last_value = ts['y'].rolling(window).mean().iloc[-1]
        future_dates = [ts['ds'].iloc[-1] + timedelta(days=i + 1) for i in range(horizon)]
        noise = np.random.normal(0, last_value * 0.05, horizon)
        trend = np.linspace(0, last_value * 0.1, horizon)
        return pd.DataFrame({
            'ds': future_dates,
            'yhat': last_value + trend + noise,
            'yhat_lower': last_value + trend + noise - last_value * 0.15,
            'yhat_upper': last_value + trend + noise + last_value * 0.15,
        })

    # ─── Prophet Forecast ────────────────────────────────────────────────────

    def _prophet_forecast(self, ts: pd.DataFrame, horizon: int = 90) -> pd.DataFrame:
        if not PROPHET_AVAILABLE:
            print("⚠️  Prophet not installed — falling back to simple moving average.")
            return self._simple_forecast(ts, horizon)
        model = Prophet(seasonality_mode='multiplicative', yearly_seasonality=True,
                        weekly_seasonality=True, daily_seasonality=False)
        model.fit(ts)
        future = model.make_future_dataframe(periods=horizon)
        forecast = model.predict(future)
        self.fitted_model = model
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(horizon)

    # ─── ARIMA Forecast ──────────────────────────────────────────────────────

    def _arima_forecast(self, ts: pd.DataFrame, horizon: int = 90) -> pd.DataFrame:
        if not ARIMA_AVAILABLE:
            print("⚠️  pmdarima not installed — falling back to simple moving average.")
            return self._simple_forecast(ts, horizon)
        model = pm.auto_arima(ts['y'], seasonal=True, m=7, stepwise=True,
                               suppress_warnings=True, error_action='ignore')
        preds, conf = model.predict(n_periods=horizon, return_conf_int=True)
        future_dates = [ts['ds'].iloc[-1] + timedelta(days=i + 1) for i in range(horizon)]
        self.fitted_model = model
        return pd.DataFrame({
            'ds': future_dates,
            'yhat': preds,
            'yhat_lower': conf[:, 0],
            'yhat_upper': conf[:, 1],
        })

    # ─── XGBoost Forecast ────────────────────────────────────────────────────

    def _xgboost_forecast(self, ts: pd.DataFrame, horizon: int = 90) -> pd.DataFrame:
        if not XGB_AVAILABLE:
            print("⚠️  XGBoost not installed — falling back to simple moving average.")
            return self._simple_forecast(ts, horizon)

        df = ts.copy()
        df['dayofweek'] = df['ds'].dt.dayofweek
        df['month'] = df['ds'].dt.month
        df['lag_7'] = df['y'].shift(7).fillna(method='bfill')
        df['lag_30'] = df['y'].shift(30).fillna(method='bfill')
        df['rolling_7'] = df['y'].rolling(7).mean().fillna(method='bfill')

        feat_cols = ['dayofweek', 'month', 'lag_7', 'lag_30', 'rolling_7']
        X, y = df[feat_cols], df['y']
        model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
        model.fit(X, y)

        future_rows = []
        last_known = df['y'].values.tolist()
        last_date = df['ds'].iloc[-1]
        for i in range(horizon):
            nd = last_date + timedelta(days=i + 1)
            lag7 = last_known[-7] if len(last_known) >= 7 else last_known[-1]
            lag30 = last_known[-30] if len(last_known) >= 30 else last_known[-1]
            roll7 = np.mean(last_known[-7:])
            row = {'dayofweek': nd.dayofweek, 'month': nd.month,
                   'lag_7': lag7, 'lag_30': lag30, 'rolling_7': roll7}
            pred = model.predict(pd.DataFrame([row]))[0]
            last_known.append(pred)
            future_rows.append({'ds': nd, 'yhat': pred,
                                 'yhat_lower': pred * 0.85, 'yhat_upper': pred * 1.15})

        self.fitted_model = model
        return pd.DataFrame(future_rows)

    # ─── Public API ──────────────────────────────────────────────────────────

    def forecast(self, transactions_df: pd.DataFrame, horizon: int = 90,
                 freq: str = "D") -> pd.DataFrame:
        """Run forecast for `horizon` days ahead."""
        ts = self.prepare_revenue_series(transactions_df, freq)
        print(f"📈 Forecasting {horizon} days — model: {self.model_type}")

        if self.model_type == 'prophet':
            self.forecast_df = self._prophet_forecast(ts, horizon)
        elif self.model_type == 'arima':
            self.forecast_df = self._arima_forecast(ts, horizon)
        elif self.model_type == 'xgboost':
            self.forecast_df = self._xgboost_forecast(ts, horizon)
        else:
            self.forecast_df = self._simple_forecast(ts, horizon)

        total = self.forecast_df['yhat'].sum()
        print(f"   Projected revenue (next {horizon}d): ${total:,.2f}")
        return self.forecast_df

    def get_forecast_summary(self) -> dict:
        """Return key forecast KPIs."""
        if self.forecast_df is None:
            return {}
        return {
            'horizon_days': len(self.forecast_df),
            'total_projected_revenue': round(self.forecast_df['yhat'].sum(), 2),
            'avg_daily_revenue': round(self.forecast_df['yhat'].mean(), 2),
            'peak_day': str(self.forecast_df.loc[self.forecast_df['yhat'].idxmax(), 'ds'].date()),
            'peak_revenue': round(self.forecast_df['yhat'].max(), 2),
            'growth_trend_pct': round(
                (self.forecast_df['yhat'].iloc[-1] - self.forecast_df['yhat'].iloc[0])
                / (self.forecast_df['yhat'].iloc[0] + 1e-9) * 100, 2
            ),
        }
