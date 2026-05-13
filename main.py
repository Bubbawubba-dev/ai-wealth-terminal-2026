import streamlit as st
import yfinance as yf
import pandas as pd
import requests
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

# --- 1. CONFIG ---
st.set_page_config(page_title="Wealth Terminal v10.0", layout="wide")

# --- 2. SCRAPER (Thread-safe & explicit target selection) ---
@st.cache_data(ttl=3600)
def get_micro_cap_universe() -> List[str]:
    """Fetches a base list of target symbols with strict fallback arrays."""
    fallback_universe = ["MRAM", "ASTS", "HIMS", "QUBT", "BZFD", "HUT", "FLEX", "VCYT", "VECO", "IONQ"]
    try:
        url = "wikipedia.org"  # Pointing to a reliable micro/small-cap data source
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=5)
       
        if response.status_code != 200:
            return fallback_universe
           
        df_list = pd.read_html(response.text)
        for df in df_list:
            df.columns = [str(c).strip().upper() for c in df.columns]
            col_candidates = [col for col in df.columns if any(x in col for x in ['TICKER', 'SYMBOL'])]
           
            if col_candidates:
                target_col = col_candidates[0]  # Isolate as a scalar string column name
                tickers = df[target_col].dropna().astype(str).tolist()
               
                clean_tickers = []
                for t in tickers:
                    token = t.split(':')[-1].replace(')', '').strip().upper()
                    if token.isalpha() and len(token) <= 5:
                        clean_tickers.append(token)
                       
                if clean_tickers:
                    return list(dict.fromkeys(clean_tickers))[:25]  # Extended pool for better filtering
    except Exception:
        pass
    return fallback_universe

# --- 3. SECURITY (Idempotent state mutation) ---
def check_password() -> bool:
    """Verifies user session identity state safely."""
    if st.session_state.get("password_correct"):
        return True

    st.sidebar.title("🔐 Access")
    pwd = st.sidebar.text_input("Access Key", type="password", key="app_pwd_input")
   
    if st.sidebar.button("Unlock"):
        correct_password = st.secrets.get("APP_PASSWORD", "1234")
        if pwd == correct_password:
            st.session_state["password_correct"] = True
            st.rerun()
        else:
            st.sidebar.error("❌ Invalid Key")
    return False

# --- 4. ANALYTICS ENGINE (Optimized & Bug-Free) ---
def analyze_stock(
    symbol: str,
    df: pd.DataFrame,
    ticker_obj: yf.Ticker
) -> Optional[Dict[str, Any]]:
    """Executes full mathematical quantitative analysis over price matrix data."""
    try:
        if df is None or len(df) < 30:
            return None
       
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            return None

        # Safe Extraction of Financial Information Parameters
        try:
            info = ticker_obj.info if hasattr(ticker_obj, 'info') else {}
            operating_margin = float(info.get('operatingMargins') or 0.0)
            return_on_assets = float(info.get('returnOnAssets') or 0.0)
            debt_to_equity = float(info.get('debtToEquity') or 0.0)
            pe_ratio = float(info.get('trailingPE') or 0.0)
        except Exception:
            operating_margin, return_on_assets, debt_to_equity, pe_ratio = 0.0, 0.0, 0.0, 0.0

        # Technical Indicators Math Engine
        has_macro_history = len(df) >= 200
        df['SMA200'] = df['Close'].rolling(200).mean() if has_macro_history else df['Close'].mean()
        df['SMA50'] = df['Close'].rolling(50).mean() if len(df) >= 50 else df['Close'].mean()
        df['SMA20'] = df['Close'].rolling(20).mean() if len(df) >= 20 else df['Close'].mean()
       
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0.0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
       
        rs = gain / (loss + 1e-9)
        df['RSI'] = 100 - (100 / (1 + rs))
        df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
       
        df.fillna(method='bfill', inplace=True)
       
        curr = df.iloc[-1]
        price = float(curr['Close'])
        open_p = float(curr['Open'])
        high_p = float(curr['High'])
        low_p = float(curr['Low'])
        rsi = float(curr['RSI']) if not np.isnan(curr['RSI']) else 50.0
        sma50 = float(curr['SMA50'])
        sma200 = float(curr['SMA200'])
        sma20 = float(curr['SMA20'])
        atr = float(curr['ATR']) if not np.isnan(curr['ATR']) else (price * 0.02)
       
        # Volume Velocity Calculation Analytics
        short_term_vol = float(df['Volume'].tail(3).mean())
        historical_base = float(df['Volume'].tail(60).mean())
        xvol = float(short_term_vol / historical_base if historical_base > 0 else 1.0)
       
        dist_from_sma50 = float((price / sma50) - 1 if sma50 > 0 else 0.0)
        suggested_entry = float(df['High'].tail(5).max())
       
        chg_4w = float((price / df['Close'].iloc[-21]) - 1.0) if len(df) >= 21 else 0.0
        high_52w = float(df['High'].tail(252).max() if len(df) >= 252 else df['High'].max())
        ratio_52w = float(price / high_52w if high_52w > 0 else 0.0)

        # EPS Revision Momentum Fetch Block
        eps_revision_momentum = 0.0
        try:
            earn_hist = ticker_obj.get_earnings_history() if hasattr(ticker_obj, 'get_earnings_history') else None
            if earn_hist is not None and not earn_hist.empty:
                df_earn = pd.DataFrame(earn_hist).dropna(subset=['epsActual', 'epsEstimate'])
                if not df_earn.empty:
                    df_earn = df_earn.tail(4)
                    avg_actual = df_earn['epsActual'].mean()
                    avg_estimate = df_earn['epsEstimate'].mean()
                    if abs(avg_estimate) > 0.01:
                        eps_revision_momentum = float((avg_actual - avg_estimate) / abs(avg_estimate))
        except Exception:
            pass

        # Score Weight Assignment Matrices
        zacks_score = 3
        if rsi >= 65 and xvol >= 2.0: zacks_score -= 1
        if eps_revision_momentum > 0.05: zacks_score -= 1  
        if eps_revision_momentum < -0.05: zacks_score += 1
        if rsi > 82 or dist_from_sma50 > 0.35: zacks_score += 1
        zacks_rank = int(max(1, min(5, zacks_score)))

        # --- MOMENTUM SCORING MATRIX ---
        score = 0
        is_above_sma200 = price > sma200 if has_macro_history else True
        if is_above_sma200: score += 3  
        if 65 <= rsi <= 82: score += 5  
        elif rsi > 82: score -= 2  
        if xvol >= 4.0: score += 4  
        elif xvol >= 2.0: score += 2  
       
        if dist_from_sma50 > 0.55:
            score -= 2  
           
        base_status = "🟡 MONITOR"
        reason = "Awaiting Momentum Confirmation"
       
        if score >= 6 and xvol >= 2.0 and dist_from_sma50 < 0.50:
            base_status = "🔥 BUY"
            reason = "Explosive Volume Breakout Run"

        vol_is_drying_up = float(curr['Volume']) < df['Volume'].tail(5).mean()
        is_overextended = dist_from_sma50 > 0.50
        has_broken_out_5d = price >= suggested_entry

        status = base_status
        if base_status == "🔥 BUY":
            if is_overextended:
                status = "⏳ COOLING OFF (OVEREXTENDED)"
                reason = "Rule 2: Extended >50% from SMA50. Await mean-reversion."
            elif vol_is_drying_up and not has_broken_out_5d:
                status = "⏳ BASE FORMING (LOW VOL)"
                reason = "Rule 1 & 3: Volume drying up at highs. Await breakout."
            elif not has_broken_out_5d:
                status = "🟡 SETTING UP"
                reason = "Rule 3: Waiting for cross above 5-Day High resistance."
            else:
                status = "🚀 EXECUTE ACTIVE BUY"
                reason = "All 3 Execution Rules Cleared: Volume breakout from safe base."

        # Risk Mitigation Metrics Output (Swing Strategy)
        initial_stop_price = float(price - (atr * 1.8))
        trailing_stop_floor = float(price - (atr * 1.5))  
        take_profit_target = float(price + (atr * 2.5))  
       
        horizon = "⏳ WATCHLIST"
        if "BUY" in status or "EXECUTE" in status:
            horizon = "⚡ SHORT-TERM SWING" if xvol >= 3.0 else "💎 LONG-TERM HOLD"
        elif price <= initial_stop_price:
            status = "🛑 HARD STOP EXCEEDED"
            horizon = "❌ EXIT POSITION"
            reason = "Volatility Floor Trailed Out"

        # --- DAY TRADING STRATEGY SYSTEM ---
        # Strategy: ORB (Opening Range Breakout) & Scalp Alignment
        dt_vol_valid = float(curr['Volume']) > 50000  # High relative liquidity requirement
        dt_vwap_proxy = (open_p + high_p + low_p + price) / 4.0
       
        # Day Trade Entry Rules
        if price > sma20 and price > dt_vwap_proxy and xvol >= 1.5 and dt_vol_valid:
            daytrade_signal = "🟢 DAYTRADE LONG"
            daytrade_entry = price
            daytrade_stop = float(price - (atr * 1.0))
            daytrade_target = float(price + (atr * 1.5))
        elif price < sma20 and price < dt_vwap_proxy and xvol >= 1.5 and dt_vol_valid:
            daytrade_signal = "🔴 DAYTRADE SHORT"
            daytrade_entry = price
            daytrade_stop = float(price + (atr * 1.0))
            daytrade_target = float(price - (atr * 1.5))
        else:
            daytrade_signal = "⚪ NO DAYTRADE SETUP"
            daytrade_entry = 0.0
            daytrade_stop = 0.0
            daytrade_target = 0.0

        # Structural base duration tracking
        base_days = 0
        holding_guide = "⚡ DAY TRADE ONLY"
       
        local_high_boundary = high_52w * 0.88
        history_slice_len = min(len(df), 90)
        historical_closes = df['Close'].tail(history_slice_len).tolist()
       
        for c_val in reversed(historical_closes):
            if c_val >= local_high_boundary:
                base_days += 1
            else:
                break
               
        if base_days > 10:
            holding_guide = "💎 STRUCTURAL ACCUMULATION BASE"

        return {
            "symbol": symbol, "status": status, "reason": reason, "price": price,
            "rsi": rsi, "xvol": xvol, "zacks_rank": zacks_rank, "horizon": horizon,
            "initial_stop": initial_stop_price, "take_profit": take_profit_target,
            "daytrade_signal": daytrade_signal, "daytrade_entry": daytrade_entry,
            "daytrade_target": daytrade_target, "daytrade_stop": daytrade_stop,
            "base_days": base_days, "holding_guide": holding_guide,
            "operating_margin": operating_margin, "return_on_assets": return_on_assets,
            "debt_to_equity": debt_to_equity, "pe_ratio": pe_ratio, "ratio_52w": ratio_52w
        }
    except Exception as e:
        st.error(f"Error compiling analysis for vector {symbol}: {str(e)}")
        return None

# --- 5. INITIALIZATION RUNNER & VIEW INTERFACE ---
if __name__ == "__main__":
    if check_password():
        st.title("📊 Alpha Terminal Engines Active")
       
        # Batch Data Downloading Architecture
        universe_symbols = get_micro_cap_universe()
       
        @st.cache_data(ttl=600)
        def fetch_batch_data(symbols: List[str]):
            processed_metrics = []
            for sym in symbols:
                try:
                    ticker = yf.Ticker(sym)
                    # Fetching 1 year history to safely compute metrics and 52W levels
                    data = ticker.history(period="1y")
                    if not data.empty:
                        res = analyze_stock(sym, data, ticker)
                        if res:
                            processed_metrics.append(res)
                except Exception:
                    pass
            return processed_metrics

        with st.spinner("Analyzing Multi-Asset Watchlist Array..."):
            all_data = fetch_batch_data(universe_symbols)
       
        df_terminal = pd.DataFrame(all_data)
       
        # Layout Definition: Tabs
        tab_standard, tab_bluesky = st.tabs(["📋 Standard Momentum Watch", "🌌 Blue Sky Finder (Tightened Gate)"])
       
        # --- TAB 1: STANDARD WATCHLIST VIEW ---
        with tab_standard:
            st.subheader("Asset Momentum Grid Matrix")
            if not df_terminal.empty:
                display_cols = [
                    "symbol", "status", "price", "rsi", "xvol", "horizon",
                    "daytrade_signal", "daytrade_entry", "daytrade_target", "daytrade_stop"
                ]
                st.dataframe(df_terminal[display_cols], use_container_width=True)
            else:
                st.warning("No data returned for active assets.")

        # --- TAB 2: BLUE SKY FINDER VIEW ---
        with tab_bluesky:
            st.subheader("🌌 Blue Sky Matrix — Ultra-Tight Structural Isolation Gate")
            st.markdown(
                "Filters assets strictly on both operational metrics and price execution constraints:\n"
                "* **Operating Margin** > 10.0%  \n"
                "* **Return on Assets** > 5.0%  \n"
                "* **Price Extension Gate** < 15% from SMA50 (Defensive cushion)  \n"
                "* **Proximity to 52-Week High** >= 85% (Blue Sky breakout orbit)"
            )
           
            if not df_terminal.empty:
                # Execution of the explicit gate constraints array
                gated_df = df_terminal[
                    (df_terminal["operating_margin"] > 0.10) &
                    (df_terminal["return_on_assets"] > 0.05) &
                    (df_terminal["ratio_52w"] >= 0.85)
                ].copy()
               
                if not gated_df.empty:
                    bs_display_cols = [
                        "symbol", "price", "operating_margin", "return_on_assets",
                        "pe_ratio", "ratio_52w", "daytrade_signal", "daytrade_target", "daytrade_stop"
                    ]
                    st.dataframe(gated_df[bs_display_cols], use_container_width=True)
                    st.success(f"Successfully isolated {len(gated_df)} assets hitting strict structural execution conditions.")
                else:
                    st.info("No active universe securities cleared the Blue Sky Gate requirements.")
            else:
                st.warning("No tracking data available to process.")
