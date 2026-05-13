import streamlit as st
import yfinance as yf
import pandas as pd
import requests
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Any, Optional

# --- 1. CONFIG ---
st.set_page_config(page_title="Wealth Terminal v13.0", layout="wide")

# --- 2. SCRAPER ---
@st.cache_data(ttl=3600)
def get_micro_cap_universe() -> List[str]:
    """Fetches a base list of target symbols with strict fallback arrays."""
    fallback_universe = ["MRAM", "ASTS", "HIMS", "QUBT", "BZFD", "HUT", "FLEX", "VCYT", "VECO", "IONQ"]
    try:
        url = "wikipedia.org"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=3)
        if response.status_code != 200:
            return fallback_universe
        df_list = pd.read_html(response.text)
        for df in df_list:
            df.columns = [str(c).strip().upper() for c in df.columns]
            col_candidates = [col for col in df.columns if any(x in col for x in ['TICKER', 'SYMBOL'])]
            if col_candidates:
                target_col = col_candidates
                tickers = df[target_col].dropna().astype(str).tolist()
                clean_tickers = []
                for t in tickers:
                    token = t.split(':')[-1].replace(')', '').strip().upper()
                    if token.isalpha() and len(token) <= 5:
                        clean_tickers.append(token)
                if clean_tickers:
                    return list(dict.fromkeys(clean_tickers))[:25]
    except Exception:
        pass
    return fallback_universe

# --- 3. SECURITY ---
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

# --- 4. OPTIMIZED ENGINE WITH MATRIX PASSTHROUGH ---
def analyze_stock_optimized(
    symbol: str,
    df: pd.DataFrame,
    spy_perf_4w: float
) -> Optional[Dict[str, Any]]:
    """Processes historical price arrays with zero trailing network requests."""
    try:
        if df is None or len(df) < 30:
            return None
           
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Vector technical processing
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
       
        short_term_vol = float(df['Volume'].tail(3).mean())
        historical_base = float(df['Volume'].tail(60).mean())
        xvol = float(short_term_vol / historical_base if historical_base > 0 else 1.0)
       
        dist_from_sma50 = float((price / sma50) - 1 if sma50 > 0 else 0.0)
        suggested_entry = float(df['High'].tail(5).max())
       
        chg_4w = float((price / df['Close'].iloc[-21]) - 1.0) if len(df) >= 21 else 0.0
        high_52w = float(df['High'].tail(252).max() if len(df) >= 252 else df['High'].max())
        ratio_52w = float(price / high_52w if high_52w > 0 else 0.0)

        alpha_spread = chg_4w - spy_perf_4w

        # --- MOMENTUM MATRIX ---
        score = 0
        if (price > sma200 if has_macro_history else True): score += 3  
        if 65 <= rsi <= 82: score += 5  
        elif rsi > 82: score -= 2  
        if xvol >= 4.0: score += 4  
        elif xvol >= 2.0: score += 2  
        if dist_from_sma50 > 0.55: score -= 2  
           
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
            if is_overextended: status = "⏳ COOLING OFF"
            elif vol_is_drying_up and not has_broken_out_5d: status = "⏳ BASE FORMING"
            elif not has_broken_out_5d: status = "🟡 SETTING UP"
            else: status = "🚀 EXECUTE ACTIVE BUY"

        initial_stop_price = float(price - (atr * 1.8))
        take_profit_target = float(price + (atr * 2.5))  

        # --- DAY TRADING STRATEGY SYSTEM ---
        dt_vol_valid = float(curr['Volume']) > 50000
        dt_vwap_proxy = (open_p + high_p + low_p + price) / 4.0
       
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
            daytrade_entry, daytrade_stop, daytrade_target = 0.0, 0.0, 0.0

        # Structural base tracking
        base_days = 0
        holding_guide = "⚡ DAY TRADE ONLY"
        local_high_boundary = high_52w * 0.88
        historical_closes = df['Close'].tail(min(len(df), 90)).tolist()
        for c_val in reversed(historical_closes):
            if c_val >= local_high_boundary: base_days += 1
            else: break
        if base_days > 10: holding_guide = "💎 STRUCTURAL BASE"

        return {
            "symbol": symbol, "status": status, "reason": reason, "price": price,
            "rsi": rsi, "xvol": xvol, "initial_stop": initial_stop_price, "take_profit": take_profit_target,
            "daytrade_signal": daytrade_signal, "daytrade_entry": daytrade_entry,
            "daytrade_target": daytrade_target, "daytrade_stop": daytrade_stop,
            "base_days": base_days, "holding_guide": holding_guide,
            "ratio_52w": ratio_52w, "chg_4w": chg_4w, "alpha_spread": alpha_spread
        }
    except Exception:
        return None

# --- 5. PLOTLY GRAPH GENERATOR ---
def create_dynamic_chart(symbol: str, df: pd.DataFrame, metrics: Dict[str, Any]) -> go.Figure:
    """Generates a high-performance 3-pane sub-plot architecture from in-memory arrays."""
    # Slicing the dataframe to show only the last 60 days for cleaner charting visibility
    chart_df = df.tail(60).copy()
    if isinstance(chart_df.columns, pd.MultiIndex):
        chart_df.columns = chart_df.columns.get_level_values(0)
       
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        vertical_spacing=0.04, row_heights=[0.6, 0.2, 0.2]
    )
   
    # Pane 1: Candlesticks & Technical Overlays
    fig.add_trace(go.Candlestick(
        x=chart_df.index, open=chart_df['Open'], high=chart_df['High'],
        low=chart_df['Low'], close=chart_df['Close'], name="Price"
    ), row=1, col=1)
   
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['SMA20'], line=dict(color='orange', width=1.5), name="SMA20"), row=1, col=1)
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['SMA50'], line=dict(color='blue', width=1.5), name="SMA50"), row=1, col=1)
   
    # Real-Time Day Trading Boundary Mark Overlays
    if metrics["daytrade_entry"] > 0:
        fig.add_hline(y=metrics["daytrade_entry"], line_dash="dash", line_color="green", annotation_text="DT Entry Trigger", row=1, col=1)
        fig.add_hline(y=metrics["daytrade_target"], line_dash="dot", line_color="cyan", annotation_text="DT Profit Target", row=1, col=1)
        fig.add_hline(y=metrics["daytrade_stop"], line_dash="dot", line_color="red", annotation_text="DT Stop Loss", row=1, col=1)

    # Pane 2: Volume Multiplier Bars
    fig.add_trace(go.Bar(x=chart_df.index, y=chart_df['Volume'], marker_color='purple', name="Volume"), row=2, col=1)
   
    # Pane 3: Relative Strength Index (RSI Metrics Grid)
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['RSI'], line=dict(color='magenta', width=1.5), name="RSI"), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
   
    # Layout Stylings
    fig.update_layout(
        title=f"🎯 Dynamic Technical Radar Matrix: {symbol} ({metrics['status']})",
        xaxis_rangeslider_visible=False, height=750, template="plotly_dark",
        margin=dict(l=50, r=50, t=50, b=50)
    )
    return fig

# --- 6. RUNNER INTERFACE ---
if __name__ == "__main__":
    if check_password():
        st.title("🚀 Lightning Fast Alpha Terminal Active")
        universe_symbols = get_micro_cap_universe()
       
        @st.cache_data(ttl=300)
        def batch_load_all_data(symbols: List[str]):
            all_tickers_to_fetch = list(set(symbols + ["SPY"]))
            raw_batch = yf.download(all_tickers_to_fetch, period="1y", group_by="ticker", progress=False)
           
            spy_perf_4w = 0.0
            if "SPY" in raw_batch:
                spy_df = raw_batch["SPY"].dropna(subset=['Close'])
                if len(spy_df) >= 21:
                    spy_perf_4w = float(spy_df['Close'].iloc[-1] / spy_df['Close'].iloc[-21]) - 1.0
           
            processed_metrics = []
            for sym in symbols:
                if sym in raw_batch and not raw_batch[sym].dropna(subset=['Close']).empty:
                    res = analyze_stock_optimized(sym, raw_batch[sym].copy(), spy_perf_4w)
                    if res: processed_metrics.append(res)
            return processed_metrics, spy_perf_4w, raw_batch

        with st.spinner("Downloading and parsing asset matrices in parallel threads..."):
            all_data, spy_benchmark_perf, raw_data_matrix = batch_load_all_data(universe_symbols)
       
        df_terminal = pd.DataFrame(all_data)
       
        tab_standard, tab_bluesky, tab_alpha, tab_visuals = st.tabs([
            "📋 Standard Momentum Watch", "🌌 Blue Sky Finder", "⚡ Alpha Engine", "📈 Interactive Plots"
        ])
       
        with tab_standard:
            st.subheader("Asset Momentum Grid Matrix")
            if not df_terminal.empty:
                st.dataframe(df_terminal[["symbol", "status", "price", "rsi", "xvol", "daytrade_signal", "daytrade_target", "daytrade_stop"]], use_container_width=True)

        with tab_bluesky:
            st.subheader("🌌 Blue Sky Matrix")
            if not df_terminal.empty:
                gated_df = df_terminal[df_terminal["ratio_52w"] >= 0.85].copy()
                if not gated_df.empty:
                    st.dataframe(gated_df[["symbol", "price", "ratio_52w", "daytrade_signal", "daytrade_target", "daytrade_stop"]], use_container_width=True)
                else:
                    st.info("No active stocks in upper breakout range.")

        with tab_alpha:
            st.subheader("⚡ Alpha Performance Arbitrage Radar")
            st.metric(label="SPY 4-Week Reference Baseline", value=f"{spy_benchmark_perf * 100:.2f}%")
            if not df_terminal.empty:
                alpha_sorted_df = df_terminal.sort_values(by="alpha_spread", ascending=False).copy()
                alpha_sorted_df["Stock 4W Perf (%)"] = alpha_sorted_df["chg_4w"] * 100.0
                alpha_sorted_df["Alpha Spread (%)"] = alpha_sorted_df["alpha_spread"] * 100.0
                st.dataframe(alpha_sorted_df[["symbol", "price", "status", "Stock 4W Perf (%)", "Alpha Spread (%)", "holding_guide"]], use_container_width=True)

        # --- TAB 4: INTERACTIVE VISUAL PLOTS (PLOTLY INTEGRATION) ---
        with tab_visuals:
            st.subheader("📈 Dynamic Analytical Charting Canvas")
            if not df_terminal.empty:
                # Let user choose which ticker to inspect interactively
                target_symbol = st.selectbox("Select Asset Signature Vector to Inspect:", df_terminal["symbol"].tolist())
               
                # Fetch local data matrix slice and metrics row
                ticker_history = raw_data_matrix[target_symbol]
                ticker_metrics = df_terminal[df_terminal["symbol"] == target_symbol].iloc[0].to_dict()
               
                # Build and display interactive charts
                chart_figure = create_dynamic_chart(target_symbol, ticker_history, ticker_metrics)
                st.plotly_chart(chart_figure, use_container_width=True)
            else:
                st.warning("No graphing vector arrays loaded.")
