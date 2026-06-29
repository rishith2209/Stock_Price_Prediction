# Stock Price Prediction Application
# Complete ML Lifecycle: Data → Preprocessing → Model → Evaluation → Deployment

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from pandas.tseries.offsets import BDay
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

# ML imports
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
from xgboost import XGBRegressor

# Technical Analysis imports
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volume import OnBalanceVolumeIndicator
from ta.volatility import BollingerBands

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Configuration
MODEL_DIR = "models"
TRADING_DAYS_THRESHOLD = 500  # 2 years ≈ 500 trading days
os.makedirs(MODEL_DIR, exist_ok=True)

# Page configuration
st.set_page_config(
    page_title="Stock Price Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Feature list (canonical)
FEATURE_COLS = [
    'MA7', 'MA21', 'MA50', 'Volatility', 'Pct_Change', 'Momentum',
    'Lag1_Close', 'Lag2_Close', 'Lag3_Close',
    'RSI', 'MACD_line', 'MACD_signal', 'OBV',
    'BB_upper', 'BB_lower', 'BB_middle', 'BB_width', 'BB_position'
]

# ==================== HELPER FUNCTIONS ====================

def flatten_multiindex_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns from yfinance into single-level names."""
    if isinstance(df.columns, pd.MultiIndex):
        new_cols = []
        for col in df.columns.values:
            parts = [str(x).strip() for x in col if str(x).strip() != ""]
            if len(parts) == 0:
                new_cols.append("")
            else:
                new_cols.append("_".join(parts))
        df.columns = new_cols
    df.columns = [str(c).strip() for c in df.columns]
    return df

def normalize_ticker_input(user_input: str, market: str) -> str:
    """
    Normalize and resolve ticker input to proper yfinance format.
    Handles: lowercase, uppercase, 6-digit BSE codes, common variations.
    """
    if not user_input or not user_input.strip():
        return ""
    
    user_input = user_input.strip()
    
    # NSE handling
    if market == "NSE":
        ticker = user_input.upper()
        # Remove .NS if already present
        if ticker.endswith('.NS'):
            ticker = ticker[:-3]
        return ticker + ".NS"
    
    # BSE handling
    elif market == "BSE":
        ticker = user_input.strip()
        # If it's a 6-digit code, use directly
        if ticker.isdigit() and len(ticker) == 6:
            return ticker + ".BO"
        # Try to resolve company name to BSE code
        ticker_upper = ticker.upper()
        # Remove .BO if already present
        if ticker_upper.endswith('.BO'):
            ticker_upper = ticker_upper[:-3]
        
        try:
            temp = yf.Ticker(ticker_upper + ".BO")
            info = {}
            try:
                info = temp.info
            except:
                pass
            
            # Check if we got a valid symbol
            scrip_code = info.get("symbol", "")
            if scrip_code and str(scrip_code).replace(".BO", "").isdigit():
                return scrip_code
            elif ticker_upper.isdigit():
                return ticker_upper + ".BO"
            else:
                return ticker_upper + ".BO"
        except:
            # Fallback: return as-is with .BO suffix
            return ticker_upper + ".BO"
    
    return user_input

def fetch_stock_data(ticker: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch OHLCV data from yfinance with robust error handling.
    """
    try:
        # Try downloading data
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        
        if df is None or df.empty:
            raise RuntimeError(f"No data returned from yfinance for {ticker}")
        
        # Handle MultiIndex columns
        df = flatten_multiindex_columns(df)
        
        # Debug: Show actual column names if needed
        # st.write(f"Debug - Actual columns: {df.columns.tolist()}")
        
        # Normalize column names (handle various yfinance versions and formats)
        col_map = {}
        for c in df.columns:
            # Convert to string and normalize
            col_str = str(c).strip()
            lc = col_str.lower().replace(" ", "").replace(".", "").replace("_", "")
            
            # Map to standard column names
            if lc == 'open':
                col_map[c] = 'Open'
            elif lc == 'high':
                col_map[c] = 'High'
            elif lc == 'low':
                col_map[c] = 'Low'
            elif lc in ['close', 'adjclose']:
                # Prioritize Close over Adj Close - only map if Close not already mapped
                if 'Close' not in col_map.values():
                    col_map[c] = 'Close'
            elif 'volume' in lc:
                col_map[c] = 'Volume'
        
        # Apply column mapping
        if col_map:
            df = df.rename(columns=col_map)
        
        # If we still don't have the columns, try alternative approach
        # Sometimes yfinance returns columns directly without MultiIndex
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing = [c for c in required if c not in df.columns]
        
        if missing:
            # Try direct access using Ticker object
            try:
                ticker_obj = yf.Ticker(ticker)
                hist = ticker_obj.history(period=period, interval=interval)
                
                if hist is not None and not hist.empty:
                    # Clean column names
                    hist.columns = [str(col).strip() for col in hist.columns]
                    
                    # Map columns
                    hist_col_map = {}
                    for col in hist.columns:
                        lc = str(col).lower().replace(" ", "").replace(".", "").replace("_", "")
                        if lc == 'open':
                            hist_col_map[col] = 'Open'
                        elif lc == 'high':
                            hist_col_map[col] = 'High'
                        elif lc == 'low':
                            hist_col_map[col] = 'Low'
                        elif lc in ['close', 'adjclose']:
                            hist_col_map[col] = 'Close'
                        elif 'volume' in lc:
                            hist_col_map[col] = 'Volume'
                    
                    if hist_col_map:
                        hist = hist.rename(columns=hist_col_map)
                    
                    # Check if we have required columns now
                    missing_after = [c for c in required if c not in hist.columns]
                    if not missing_after:
                        df = hist
                    else:
                        raise RuntimeError(f"Missing required columns after alternative fetch: {missing_after}. Available columns: {hist.columns.tolist()}")
                else:
                    raise RuntimeError(f"Alternative fetch returned empty data")
            except Exception as alt_e:
                raise RuntimeError(f"Missing required columns: {missing}. Available columns: {df.columns.tolist()}. Alternative fetch also failed: {str(alt_e)}")
        
        # Select only required columns
        df = df[required].copy()
        
        # Drop rows with any NaN values
        df = df.dropna(how='any')
        
        if df.empty:
            raise RuntimeError("No data remaining after filtering NaN values")
        
        # Ensure columns are strings
        df.columns = [str(c).strip() for c in df.columns]
        
        return df
        
    except Exception as e:
        raise RuntimeError(f"Error fetching data for {ticker}: {str(e)}")

def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute comprehensive technical indicators."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']
    
    # Moving Averages
    df['MA7'] = close.rolling(7, min_periods=1).mean()
    df['MA21'] = close.rolling(21, min_periods=1).mean()
    df['MA50'] = close.rolling(50, min_periods=1).mean()
    
    # Volatility and Momentum
    df['Volatility'] = close.rolling(20, min_periods=1).std().fillna(0)
    df['Pct_Change'] = close.pct_change().fillna(0)
    df['Momentum'] = (close - close.shift(10)).fillna(0)
    
    # Lag features
    for lag in [1, 2, 3]:
        if len(close) > 0:
            df[f'Lag{lag}_Close'] = close.shift(lag).bfill().fillna(close.iloc[0])
        else:
            df[f'Lag{lag}_Close'] = 0
    
    # RSI
    try:
        rsi = RSIIndicator(close=close, window=14)
        df['RSI'] = rsi.rsi().bfill().fillna(50)  # Default to neutral RSI
    except:
        df['RSI'] = 50
    
    # MACD
    try:
        macd = MACD(close=close)
        df['MACD_line'] = macd.macd().bfill().fillna(0)
        df['MACD_signal'] = macd.macd_signal().bfill().fillna(0)
    except:
        df['MACD_line'] = 0
        df['MACD_signal'] = 0
    
    # OBV (On-Balance Volume)
    try:
        obv = OnBalanceVolumeIndicator(close=close, volume=volume)
        df['OBV'] = obv.on_balance_volume().bfill().fillna(0)
    except:
        df['OBV'] = 0
    
    # Bollinger Bands
    try:
        bb = BollingerBands(close=close, window=20, window_dev=2)
        df['BB_upper'] = bb.bollinger_hband().bfill().fillna(close)
        df['BB_lower'] = bb.bollinger_lband().bfill().fillna(close)
        df['BB_middle'] = bb.bollinger_mavg().bfill().fillna(close)
        # Additional BB features
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
        df['BB_position'] = (close - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
        df['BB_width'] = df['BB_width'].fillna(0)
        df['BB_position'] = df['BB_position'].fillna(0.5)
    except:
        df['BB_upper'] = close
        df['BB_lower'] = close
        df['BB_middle'] = close
        df['BB_width'] = 0
        df['BB_position'] = 0.5
    
    df.columns = df.columns.str.strip()
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features and create target variables for next day prediction."""
    base = compute_technical_indicators(df)
    data = base.copy()
    data.columns = data.columns.str.strip()
    
    # Create target variables (next day's values)
    data['Target_Close'] = data['Close'].shift(-1)
    data['Target_High'] = data['High'].shift(-1)
    data['Target_Low'] = data['Low'].shift(-1)
    
    # Remove rows with NaN targets (last row)
    data = data.dropna(subset=['Target_Close', 'Target_High', 'Target_Low'])
    
    if len(data) < 100:
        raise ValueError(f"Insufficient data after feature engineering: {len(data)} rows (< 100 required)")
    
    data.columns = data.columns.str.strip()
    return data

# ==================== MODEL FUNCTIONS ====================

def get_trading_days_count(df: pd.DataFrame) -> int:
    """Count trading days in the dataset."""
    try:
        if isinstance(df.index[0], pd.Timestamp):
            days = (df.index[-1] - df.index[0]).days
            # Approximate trading days (excluding weekends, roughly 252 per year)
            trading_days = len(df)
            return trading_days
        return len(df)
    except:
        return len(df)

def select_model_type(trading_days: int) -> str:
    """Select model based on data availability."""
    if trading_days < TRADING_DAYS_THRESHOLD:
        return "RandomForest"
    else:
        return "XGBoost"

def build_model(model_name: str, hyperparameters: dict = None):
    """Build model with specified hyperparameters."""
    if model_name == "RandomForest":
        params = {
            'random_state': 42,
            'n_jobs': -1
        }
        if hyperparameters:
            params.update(hyperparameters)
        return RandomForestRegressor(**params)
    
    elif model_name == "XGBoost":
        params = {
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0
        }
        if hyperparameters:
            params.update(hyperparameters)
        return XGBRegressor(**params)
    
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def optimize_hyperparameters(X: pd.DataFrame, y: pd.Series, model_name: str) -> dict:
    """Systematically optimize hyperparameters using GridSearchCV."""
    if model_name == "RandomForest":
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 7, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        base_model = RandomForestRegressor(random_state=42, n_jobs=1)
    
    elif model_name == "XGBoost":
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 0.9, 1.0]
        }
        base_model = XGBRegressor(random_state=42, n_jobs=1, verbosity=0)
    
    else:
        return {}
    
    # Use TimeSeriesSplit for cross-validation
    tscv = TimeSeriesSplit(n_splits=3)
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=tscv,
        scoring='neg_mean_squared_error',
        n_jobs=1,
        verbose=0
    )
    
    grid_search.fit(X, y)
    return grid_search.best_params_

def evaluate_model(X: pd.DataFrame, y: pd.Series, model, scaler=None, model_name: str = ""):
    """Evaluate model using walk-forward validation."""
    tscv = TimeSeriesSplit(n_splits=5)
    metrics_list = {'rmse': [], 'mae': [], 'mape': [], 'r2': []}
    
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Scale if needed
        if scaler is not None:
            X_train_scaled = pd.DataFrame(
                scaler.fit_transform(X_train),
                index=X_train.index,
                columns=X_train.columns
            )
            X_test_scaled = pd.DataFrame(
                scaler.transform(X_test),
                index=X_test.index,
                columns=X_test.columns
            )
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test
        
        # Train model
        model_copy = build_model(model_name, model.get_params() if hasattr(model, 'get_params') else {})
        model_copy.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = model_copy.predict(X_test_scaled)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        # Calculate MAPE manually to handle zero values
        non_zero_mask = y_test != 0
        if non_zero_mask.any():
            mape = np.mean(np.abs((y_test[non_zero_mask] - y_pred[non_zero_mask]) / y_test[non_zero_mask])) * 100
        else:
            mape = np.inf
        r2 = r2_score(y_test, y_pred)
        
        metrics_list['rmse'].append(rmse)
        metrics_list['mae'].append(mae)
        metrics_list['mape'].append(mape)
        metrics_list['r2'].append(r2)
    
    return {
        'rmse': float(np.mean(metrics_list['rmse'])),
        'mae': float(np.mean(metrics_list['mae'])),
        'mape': float(np.mean(metrics_list['mape'])),
        'r2': float(np.mean(metrics_list['r2'])),
        'rmse_std': float(np.std(metrics_list['rmse'])),
        'mae_std': float(np.std(metrics_list['mae']))
    }

def train_models(X: pd.DataFrame, y_dict: dict, model_name: str, optimize: bool = False, use_pca: bool = False, n_components: float = 0.95):
    """
    Train models for Close, High, and Low predictions.
    
    Parameters:
    - use_pca: Whether to apply PCA dimensionality reduction
    - n_components: Number of components (int) or variance to retain (float 0-1)
    """
    models = {}
    scalers = {}
    pca_transformers = {}
    metrics = {}
    
    # Determine feature order
    feature_order = [c for c in FEATURE_COLS if c in X.columns]
    missing_cols = [c for c in X.columns if c not in feature_order]
    feature_order = feature_order + sorted(missing_cols)
    X_original = X[feature_order].copy()
    
    # Apply PCA if requested
    if use_pca:
        # Standardize features before PCA
        scaler_pre_pca = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler_pre_pca.fit_transform(X_original),
            index=X_original.index,
            columns=X_original.columns
        )
        
        # Apply PCA
        pca = PCA(n_components=n_components)
        X_transformed = pca.fit_transform(X_scaled)
        
        # Create DataFrame with principal components
        n_comp_actual = X_transformed.shape[1]
        X_transformed_df = pd.DataFrame(
            X_transformed,
            index=X_original.index,
            columns=[f'PC{i+1}' for i in range(n_comp_actual)]
        )
        
        # Store explained variance info
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        st.info(f"PCA applied: {len(feature_order)} features → {n_comp_actual} components "
                f"({cumulative_variance[-1]*100:.2f}% variance retained)")
        
        X_final = X_transformed_df
        feature_order_pca = list(X_transformed_df.columns)
    else:
        X_final = X_original
        feature_order_pca = feature_order
        scaler_pre_pca = None
        pca = None
    
    for target_name, y in y_dict.items():
        # Optimize hyperparameters if requested
        best_params = {}
        if optimize:
            with st.spinner(f"Optimizing hyperparameters for {target_name}..."):
                best_params = optimize_hyperparameters(X_final, y, model_name)
        
        # Build and train model
        model = build_model(model_name, best_params)
        scaler = scaler_pre_pca  # Use pre-PCA scaler if PCA was applied
        
        # Train model
        model.fit(X_final, y)
        
        # Evaluate model
        eval_metrics = evaluate_model(X_final, y, model, scaler, model_name)
        metrics[target_name] = eval_metrics
        
        models[target_name] = model
        scalers[target_name] = scaler
        if use_pca:
            pca_transformers[target_name] = pca
    
    if use_pca:
        return models, scalers, metrics, feature_order_pca, pca_transformers, feature_order
    else:
        return models, scalers, metrics, feature_order_pca, None, None

def predict_next_day(df: pd.DataFrame, models: dict, scalers: dict, feature_order: list, 
                     pca_transformers: dict = None, original_feature_order: list = None) -> dict:
    """
    Predict next day's High, Low, and Close prices.
    
    Parameters:
    - pca_transformers: Dictionary of PCA transformers if PCA was used
    - original_feature_order: Original feature order before PCA transformation
    """
    # Validate input
    if df is None or len(df) == 0:
        raise ValueError("DataFrame is empty or None")
    
    # Compute indicators for latest data
    df_with_indicators = compute_technical_indicators(df)
    
    if len(df_with_indicators) == 0:
        raise ValueError("No data after computing technical indicators")
    
    # Get latest row with original features
    try:
        if original_feature_order:
            latest_features_original = df_with_indicators.iloc[[-1]][original_feature_order].copy()
        else:
            latest_features_original = df_with_indicators.iloc[[-1]][feature_order].copy()
    except KeyError as e:
        raise ValueError(f"Missing features in data: {e}. Available: {df_with_indicators.columns.tolist()}")
    
    # Fill any missing values
    latest_features_original = latest_features_original.fillna(0)
    
    predictions = {}
    for target_name in ['Close', 'High', 'Low']:
        model = models[target_name]
        scaler = scalers.get(target_name)
        pca = pca_transformers.get(target_name) if pca_transformers else None
        
        X_pred = latest_features_original.copy()
        
        # Apply scaling if PCA was used
        if scaler is not None:
            X_pred = pd.DataFrame(
                scaler.transform(X_pred),
                index=X_pred.index,
                columns=X_pred.columns
            )
        
        # Apply PCA transformation if used
        if pca is not None:
            X_pred_transformed = pca.transform(X_pred.values)
            X_pred = pd.DataFrame(
                X_pred_transformed,
                index=X_pred.index,
                columns=feature_order
            )
        
        pred = model.predict(X_pred)[0]
        predictions[target_name] = float(pred)
    
    # Ensure High >= Close >= Low
    predictions['High'] = max(predictions['High'], predictions['Close'])
    predictions['Low'] = min(predictions['Low'], predictions['Close'])
    if predictions['High'] < predictions['Low']:
        predictions['High'], predictions['Low'] = predictions['Low'], predictions['High']
    
    return predictions

# ==================== VISUALIZATION FUNCTIONS ====================

def plot_actual_vs_predicted(y_actual: pd.Series, y_predicted: pd.Series, title: str = "Actual vs Predicted"):
    """Create actual vs predicted plot."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=y_actual.index,
        y=y_actual.values,
        mode='lines',
        name='Actual',
        line=dict(color='#1f77b4', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=y_predicted.index,
        y=y_predicted.values,
        mode='lines',
        name='Predicted',
        line=dict(color='#ff7f0e', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Price',
        hovermode='x unified',
        height=400,
        template='plotly_white'
    )
    
    return fig

def create_prediction_plot(df: pd.DataFrame, predictions: dict, next_date: pd.Timestamp):
    """Create comprehensive prediction visualization."""
    # Show last 60 days + prediction
    recent_df = df.tail(60).copy()
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=('Price Trend with Prediction', 'Volume'),
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3]
    )
    
    # Price chart
    fig.add_trace(go.Scatter(
        x=recent_df.index,
        y=recent_df['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='#2E86AB', width=2)
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=recent_df.index,
        y=recent_df['High'],
        mode='lines',
        name='High',
        line=dict(color='#A23B72', width=1, dash='dot'),
        opacity=0.6
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=recent_df.index,
        y=recent_df['Low'],
        mode='lines',
        name='Low',
        line=dict(color='#F18F01', width=1, dash='dot'),
        opacity=0.6
    ), row=1, col=1)
    
    # Prediction markers
    fig.add_trace(go.Scatter(
        x=[next_date],
        y=[predictions['Close']],
        mode='markers+text',
        name='Predicted Close',
        marker=dict(size=15, color='#06A77D', symbol='star'),
        text=['Pred Close'],
        textposition='top center'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=[next_date, next_date],
        y=[predictions['Low'], predictions['High']],
        mode='lines+markers',
        name='Predicted Range',
        line=dict(color='#06A77D', width=3),
        marker=dict(size=10, color='#06A77D')
    ), row=1, col=1)
    
    # Volume chart
    fig.add_trace(go.Bar(
        x=recent_df.index,
        y=recent_df['Volume'],
        name='Volume',
        marker_color='#6C757D'
    ), row=2, col=1)
    
    fig.update_layout(
        height=700,
        showlegend=True,
        template='plotly_white',
        hovermode='x unified'
    )
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    return fig

# ==================== EDA FUNCTIONS ====================

def perform_eda(df: pd.DataFrame):
    """Perform comprehensive Exploratory Data Analysis."""
    eda_results = {}
    
    # Basic statistics
    eda_results['basic_stats'] = df[['Open', 'High', 'Low', 'Close', 'Volume']].describe()
    
    # Missing values
    eda_results['missing'] = df.isnull().sum()
    
    # Returns analysis
    df_eda = df.copy()
    df_eda['Returns'] = df_eda['Close'].pct_change()
    df_eda['Log_Returns'] = np.log(df_eda['Close'] / df_eda['Close'].shift(1))
    
    eda_results['returns_stats'] = df_eda['Returns'].describe()
    eda_results['volatility'] = df_eda['Returns'].std() * np.sqrt(252)  # Annualized
    
    # Correlation matrix
    eda_results['correlation'] = df[['Open', 'High', 'Low', 'Close', 'Volume']].corr()
    
    return eda_results, df_eda

# ==================== MAIN PIPELINE ====================

@st.cache_data(ttl=3600)
def load_company_info(ticker: str):
    """Load company information with caching."""
    try:
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info
        return {
            'name': info.get('longName', info.get('shortName', ticker)),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'market_cap': info.get('marketCap', 0),
            'current_price': info.get('currentPrice', 0),
            'previous_close': info.get('previousClose', 0),
            'volume': info.get('volume', 0),
            'avg_volume': info.get('averageVolume', 0),
            '52_week_high': info.get('fiftyTwoWeekHigh', 0),
            '52_week_low': info.get('fiftyTwoWeekLow', 0)
        }
    except:
        return None

def main():
    st.title("Stock Price Prediction System")
    st.markdown("**Predict next trading day's High, Low, and Closing prices using Machine Learning**")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        market = st.selectbox("Select Market", ['NSE', 'BSE'], index=0)
        ticker_input = st.text_input("Enter Ticker Symbol", value="RELIANCE", help="Enter stock ticker (e.g., RELIANCE, INFY, or 6-digit BSE code)")
        
        st.markdown("---")
        st.header("Model Options")
        optimize_params = st.checkbox("Enable Hyperparameter Optimization", value=False, help="Takes longer but may improve accuracy")
        use_pca = st.checkbox("Apply PCA Dimensionality Reduction", value=False, help="Optional: Reduce features using Principal Component Analysis")
        show_eda = st.checkbox("Show Detailed EDA", value=True)
        show_evaluation = st.checkbox("Show Model Evaluation", value=True)
        
        # PCA component selection
        if use_pca:
            pca_variance = st.slider("Variance to Retain (%)", min_value=80, max_value=99, value=95, step=1, help="Percentage of variance to retain in PCA")
            n_components_pca = pca_variance / 100.0
        else:
            n_components_pca = 0.95
        
        st.markdown("---")
        predict_button = st.button("Generate Prediction", type="primary", use_container_width=True)
    
    if predict_button:
        if not ticker_input or not ticker_input.strip():
            st.error("Please enter a ticker symbol")
            return
        
        # Normalize ticker
        try:
            ticker = normalize_ticker_input(ticker_input, market)
            st.info(f"Resolved ticker: {ticker}")
        except Exception as e:
            st.error(f"Error resolving ticker: {e}")
            return
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Fetch data
        status_text.text("Step 1/6: Fetching stock data...")
        progress_bar.progress(10)
        try:
            df = fetch_stock_data(ticker, period="5y")
            if df.empty:
                st.error("No data retrieved for this ticker")
                return
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return
        
        # Step 2: Load company info
        status_text.text("Step 2/6: Loading company information...")
        progress_bar.progress(20)
        company_info = load_company_info(ticker)
        
        # Display company information
        if company_info:
            st.subheader("Company Information")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Company", company_info['name'])
                st.metric("Sector", company_info['sector'])
            with col2:
                if len(df) > 0 and 'Close' in df.columns:
                    current = company_info['current_price'] or df['Close'].iloc[-1]
                    prev_close = company_info['previous_close'] or (df['Close'].iloc[-2] if len(df) > 1 else current)
                else:
                    current = company_info['current_price'] or 0
                    prev_close = company_info['previous_close'] or current
                change = current - prev_close
                change_pct = (change / prev_close * 100) if prev_close > 0 and prev_close != 0 else 0
                st.metric("Current Price", f"₹{current:.2f}", f"{change_pct:+.2f}%")
            with col3:
                st.metric("52W High", f"₹{company_info['52_week_high']:.2f}" if company_info['52_week_high'] else "N/A")
                st.metric("52W Low", f"₹{company_info['52_week_low']:.2f}" if company_info['52_week_low'] else "N/A")
            with col4:
                st.metric("Industry", company_info['industry'])
        
        st.markdown("---")
        
        # Step 3: EDA
        status_text.text("Step 3/6: Performing Exploratory Data Analysis...")
        progress_bar.progress(30)
        
        if show_eda:
            st.subheader("Exploratory Data Analysis")
            
            # Basic info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Trading Days", len(df))
            with col2:
                date_range = f"{df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}"
                st.metric("Date Range", date_range)
            with col3:
                trading_days = get_trading_days_count(df)
                st.metric("Trading Days Available", trading_days)
            
            # Perform EDA
            eda_results, df_eda = perform_eda(df)
            
            # Missing values
            st.markdown("#### Missing Values Analysis")
            missing_df = pd.DataFrame({
                'Column': eda_results['missing'].index,
                'Missing Count': eda_results['missing'].values,
                'Missing %': (eda_results['missing'].values / len(df) * 100)
            })
            st.dataframe(missing_df, use_container_width=True)
            
            # Statistical summary
            st.markdown("#### Statistical Summary")
            st.dataframe(eda_results['basic_stats'], use_container_width=True)
            
            # Returns analysis
            st.markdown("#### Returns Analysis")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Annualized Volatility", f"{eda_results['volatility']*100:.2f}%")
                st.metric("Mean Daily Return", f"{eda_results['returns_stats']['mean']*100:.4f}%")
            with col2:
                st.metric("Max Daily Return", f"{eda_results['returns_stats']['max']*100:.2f}%")
                st.metric("Min Daily Return", f"{eda_results['returns_stats']['min']*100:.2f}%")
            
            # Visualizations
            tab1, tab2, tab3, tab4 = st.tabs(["Price Trends", "Returns Distribution", "Correlation Matrix", "Autocorrelation"])
            
            with tab1:
                fig_price = go.Figure()
                fig_price.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close', line=dict(color='#2E86AB')))
                fig_price.add_trace(go.Scatter(x=df.index, y=df['High'], mode='lines', name='High', opacity=0.5, line=dict(color='#A23B72')))
                fig_price.add_trace(go.Scatter(x=df.index, y=df['Low'], mode='lines', name='Low', opacity=0.5, line=dict(color='#F18F01')))
                fig_price.update_layout(title="Historical Price Trend", xaxis_title="Date", yaxis_title="Price", template='plotly_white', height=400)
                st.plotly_chart(fig_price, use_container_width=True)
            
            with tab2:
                fig_returns = go.Figure()
                fig_returns.add_trace(go.Histogram(x=df_eda['Returns'].dropna(), nbinsx=50, name='Returns', marker_color='#06A77D'))
                fig_returns.update_layout(title="Returns Distribution", xaxis_title="Daily Return", yaxis_title="Frequency", template='plotly_white', height=400)
                st.plotly_chart(fig_returns, use_container_width=True)
            
            with tab3:
                fig_corr = go.Figure(data=go.Heatmap(
                    z=eda_results['correlation'].values,
                    x=eda_results['correlation'].columns,
                    y=eda_results['correlation'].index,
                    colorscale='RdBu',
                    zmid=0,
                    text=eda_results['correlation'].round(2).values,
                    texttemplate='%{text}',
                    textfont={"size":10}
                ))
                fig_corr.update_layout(title="Correlation Matrix", template='plotly_white', height=400)
                st.plotly_chart(fig_corr, use_container_width=True)
            
            with tab4:
                fig_acf = plt.figure(figsize=(10, 4))
                plot_acf(df_eda['Returns'].dropna(), lags=30, ax=plt.gca())
                plt.title("Autocorrelation of Returns")
                st.pyplot(fig_acf)
            
            st.markdown("---")
        
        # Step 4: Feature Engineering
        status_text.text("Step 4/6: Engineering features...")
        progress_bar.progress(40)
        
        try:
            data = engineer_features(df)
            X = data[[c for c in FEATURE_COLS if c in data.columns]].copy()
            y_dict = {
                'Close': data['Target_Close'],
                'High': data['Target_High'],
                'Low': data['Target_Low']
            }
        except Exception as e:
            st.error(f"Error in feature engineering: {e}")
            return
        
        # Step 5: Model Selection and Training
        status_text.text("Step 5/6: Training model...")
        progress_bar.progress(50)
        
        trading_days = get_trading_days_count(df)
        model_name = select_model_type(trading_days)
        
        st.subheader("Model Information")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Trading Days Available", trading_days)
            st.metric("Selected Model", model_name)
        with col2:
            threshold_info = f"< {TRADING_DAYS_THRESHOLD} days → RandomForest"
            st.info(f"**Selection Rule:** {threshold_info}\n\n≥ {TRADING_DAYS_THRESHOLD} days → XGBoost")
        
        # Train models
        try:
            models, scalers, metrics, feature_order, pca_transformers, original_feature_order = train_models(
                X, y_dict, model_name, optimize=optimize_params, 
                use_pca=use_pca, n_components=n_components_pca
            )
        except Exception as e:
            st.error(f"Error training models: {e}")
            return
        
        # Step 6: Evaluation
        if show_evaluation:
            status_text.text("Step 6/6: Evaluating models...")
            progress_bar.progress(70)
            
            st.subheader("Model Performance Metrics")
            
            for target_name, metric_dict in metrics.items():
                with st.expander(f"{target_name} Price Model Performance"):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("RMSE", f"₹{metric_dict['rmse']:.2f}", help="Root Mean Squared Error")
                    with col2:
                        st.metric("MAE", f"₹{metric_dict['mae']:.2f}", help="Mean Absolute Error")
                    with col3:
                        st.metric("MAPE", f"{metric_dict['mape']:.2f}%", help="Mean Absolute Percentage Error")
                    with col4:
                        st.metric("R² Score", f"{metric_dict['r2']:.4f}", help="Coefficient of Determination")
                    
                    # Actual vs Predicted plot for evaluation
                    # Use last 20% of data for visualization
                    split_idx = int(len(X) * 0.8)
                    X_eval = X.iloc[split_idx:].copy()
                    y_eval_actual = y_dict[target_name].iloc[split_idx:]
                    model_eval = models[target_name]
                    y_eval_pred = model_eval.predict(X_eval)
                    y_eval_pred_series = pd.Series(y_eval_pred, index=y_eval_actual.index)
                    
                    fig_eval = plot_actual_vs_predicted(
                        y_eval_actual,
                        y_eval_pred_series,
                        f"{target_name} Price: Actual vs Predicted (Holdout Set)"
                    )
                    st.plotly_chart(fig_eval, use_container_width=True)
        
        # Prediction
        status_text.text("Generating prediction for next trading day...")
        progress_bar.progress(90)
        
        # Calculate next trading day
        if len(df) == 0:
            st.error("No data available for prediction")
            return
        
        last_date = df.index[-1]
        next_date = last_date + BDay(1)
        
        # Make prediction
        try:
            predictions = predict_next_day(
                df, models, scalers, feature_order,
                pca_transformers=pca_transformers,
                original_feature_order=original_feature_order if original_feature_order is not None else feature_order
            )
        except Exception as e:
            st.error(f"Error generating prediction: {e}")
            import traceback
            st.error(f"Details: {traceback.format_exc()}")
            return
        
        progress_bar.progress(100)
        status_text.text("Complete!")
        
        st.markdown("---")
        st.subheader("Next Trading Day Prediction")
        st.info(f"Predicted prices for **{next_date.strftime('%B %d, %Y')}** ({next_date.strftime('%A')})")
        
        # Calculate prediction accuracy metrics from model performance
        close_accuracy = (1 - metrics['Close']['mape'] / 100) * 100 if metrics['Close']['mape'] < 100 else 0
        close_r2 = metrics['Close']['r2'] * 100
        close_rmse = metrics['Close']['rmse']
        
        high_accuracy = (1 - metrics['High']['mape'] / 100) * 100 if metrics['High']['mape'] < 100 else 0
        high_r2 = metrics['High']['r2'] * 100
        high_rmse = metrics['High']['rmse']
        
        low_accuracy = (1 - metrics['Low']['mape'] / 100) * 100 if metrics['Low']['mape'] < 100 else 0
        low_r2 = metrics['Low']['r2'] * 100
        low_rmse = metrics['Low']['rmse']
        
        # Display predictions with accuracy
        col1, col2, col3 = st.columns(3)
        if len(df) > 0 and 'Close' in df.columns:
            current_close = df['Close'].iloc[-1]
            change = predictions['Close'] - current_close
            change_pct = (change / current_close * 100) if current_close > 0 and current_close != 0 else 0
        else:
            current_close = 0
            change = 0
            change_pct = 0
        
        with col1:
            st.metric("Predicted Close", f"₹{predictions['Close']:.2f}", f"{change_pct:+.2f}%")
            # Show accuracy information
            st.caption(f"Accuracy: {close_accuracy:.2f}% (based on MAPE)")
            st.caption(f"R² Score: {close_r2:.2f}%")
            st.caption(f"Expected Error: ±₹{close_rmse:.2f} (RMSE)")
        
        with col2:
            st.metric("Predicted High", f"₹{predictions['High']:.2f}")
            st.caption(f"Accuracy: {high_accuracy:.2f}% (based on MAPE)")
            st.caption(f"R² Score: {high_r2:.2f}%")
            st.caption(f"Expected Error: ±₹{high_rmse:.2f} (RMSE)")
        
        with col3:
            st.metric("Predicted Low", f"₹{predictions['Low']:.2f}")
            st.caption(f"Accuracy: {low_accuracy:.2f}% (based on MAPE)")
            st.caption(f"R² Score: {low_r2:.2f}%")
            st.caption(f"Expected Error: ±₹{low_rmse:.2f} (RMSE)")
        
        # Overall prediction confidence summary
        avg_accuracy = (close_accuracy + high_accuracy + low_accuracy) / 3
        avg_r2 = (close_r2 + high_r2 + low_r2) / 3
        avg_rmse = (close_rmse + high_rmse + low_rmse) / 3
        
        st.markdown("---")
        st.markdown("#### Prediction Accuracy Summary")
        summary_col1, summary_col2, summary_col3 = st.columns(3)
        with summary_col1:
            st.metric("Average Accuracy", f"{avg_accuracy:.2f}%", 
                     help="Average prediction accuracy based on MAPE across all three targets")
        with summary_col2:
            st.metric("Average R² Score", f"{avg_r2:.2f}%",
                     help="Average variance explained (R²) across all models")
        with summary_col3:
            st.metric("Average Expected Error", f"±₹{avg_rmse:.2f}",
                     help="Average Root Mean Squared Error across all predictions")
        
        # Accuracy interpretation
        if avg_accuracy >= 90:
            accuracy_status = "Excellent"
            accuracy_desc = "Model shows excellent predictive accuracy with minimal expected error."
            status_color = "success"
        elif avg_accuracy >= 80:
            accuracy_status = "Good"
            accuracy_desc = "Model shows good predictive accuracy. Predictions are reliable for decision-making."
            status_color = "info"
        elif avg_accuracy >= 70:
            accuracy_status = "Moderate"
            accuracy_desc = "Model shows moderate accuracy. Use predictions with caution and consider market volatility."
            status_color = "warning"
        else:
            accuracy_status = "Low"
            accuracy_desc = "Model accuracy is below optimal. Predictions should be used cautiously."
            status_color = "error"
        
        if status_color == "success":
            st.success(f"**Accuracy Status: {accuracy_status}** - {accuracy_desc} The model has been evaluated using walk-forward cross-validation on historical data.")
        elif status_color == "info":
            st.info(f"**Accuracy Status: {accuracy_status}** - {accuracy_desc} The model has been evaluated using walk-forward cross-validation on historical data.")
        elif status_color == "warning":
            st.warning(f"**Accuracy Status: {accuracy_status}** - {accuracy_desc} The model has been evaluated using walk-forward cross-validation on historical data.")
        else:
            st.error(f"**Accuracy Status: {accuracy_status}** - {accuracy_desc} The model has been evaluated using walk-forward cross-validation on historical data.")
        
        # Prediction visualization
        fig_pred = create_prediction_plot(df, predictions, next_date)
        st.plotly_chart(fig_pred, use_container_width=True)
        
        # Feature importance
        st.subheader("Feature Importance")
        try:
            model_close = models['Close']
            if hasattr(model_close, 'feature_importances_'):
                if use_pca:
                    # Show PCA component importance
                    feature_importance = pd.DataFrame({
                        'Component': feature_order,
                        'Importance': model_close.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    fig_importance = go.Figure(data=go.Bar(
                        x=feature_importance['Importance'],
                        y=feature_importance['Component'],
                        orientation='h',
                        marker_color='#2E86AB'
                    ))
                    fig_importance.update_layout(
                        title="Principal Component Importance (Close Price Model)",
                        xaxis_title="Importance",
                        yaxis_title="Principal Component",
                        template='plotly_white',
                        height=400
                    )
                    st.plotly_chart(fig_importance, use_container_width=True)
                    
                    # Show explained variance for PCA
                    if pca_transformers and 'Close' in pca_transformers:
                        pca_close = pca_transformers['Close']
                        explained_var = pca_close.explained_variance_ratio_
                        cum_var = np.cumsum(explained_var)
                        
                        fig_pca = go.Figure()
                        fig_pca.add_trace(go.Bar(
                            x=[f'PC{i+1}' for i in range(len(explained_var))],
                            y=explained_var * 100,
                            name='Individual Variance',
                            marker_color='#06A77D'
                        ))
                        fig_pca.add_trace(go.Scatter(
                            x=[f'PC{i+1}' for i in range(len(cum_var))],
                            y=cum_var * 100,
                            name='Cumulative Variance',
                            mode='lines+markers',
                            line=dict(color='#A23B72', width=2),
                            marker=dict(size=8)
                        ))
                        fig_pca.update_layout(
                            title="PCA Explained Variance",
                            xaxis_title="Principal Component",
                            yaxis_title="Variance Explained (%)",
                            template='plotly_white',
                            height=400
                        )
                        st.plotly_chart(fig_pca, use_container_width=True)
                else:
                    # Show original feature importance
                    feature_importance = pd.DataFrame({
                        'Feature': feature_order,
                        'Importance': model_close.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    fig_importance = go.Figure(data=go.Bar(
                        x=feature_importance['Importance'],
                        y=feature_importance['Feature'],
                        orientation='h',
                        marker_color='#2E86AB'
                    ))
                    fig_importance.update_layout(
                        title="Feature Importance (Close Price Model)",
                        xaxis_title="Importance",
                        yaxis_title="Feature",
                        template='plotly_white',
                        height=400
                    )
                    st.plotly_chart(fig_importance, use_container_width=True)
            else:
                st.info("Feature importance not available for this model type")
        except Exception as e:
            st.warning(f"Could not compute feature importance: {e}")
        
        st.markdown("---")
        st.success("Prediction pipeline completed successfully!")
        st.caption("**Disclaimer:** This prediction is for educational purposes only and should not be used as financial advice. Past performance does not guarantee future results.")

if __name__ == "__main__":
    main()
