import streamlit as st
import yfinance as yf
import pandas as pd
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timezone

# --- 1. CONFIG ---
st.set_page_config(page_title="Wealth Terminal v5.3", layout="wide")

# --- 2. SCRAPER ---
def get_hot_picks():
    try:
        url = "https://yahoo.com"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers, timeout=10)
        df_list = pd.read_html(response.text)
        if df_list:
            df = df_list[0]
            if 'Symbol' in df.columns:
                return df['Symbol'].dropna().tolist()[:15]
    except:
        pass
    # Fallback to alpha-growth momentum market leaders
    return ["HUT", "AMD", "SMCI", "FLEX", "AAPL", "VCYT", "VECO", "ARM", "IONQ", "PLTR"]

# --- 3. SECURITY ---
def check_password():
    if "password_correct" not in st.session_state:
        st.sidebar.title("🔐 Access")
        pwd = st.sidebar.text_input("Access Key", type="password")
        if st.sidebar.button("Unlock"):
            if pwd == st.secrets.get("APP_PASSWORD", "1234"):
                st.session_state["password_correct"] = True
                st.rerun()
            else:
                st.sidebar.error("❌ Invalid")
        return False
    return True

# --- 4. ANALYTICS ENGINE (HORIZON CLASSIFICATION SPECIFICATION) ---
def analyze_stock(symbol, df, ticker_obj, funds, risk, enable_analyst_picks):
    try:
        if df is None or len(df) < 30:
            return None
       
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            return None

        # Math Engine
        has_macro_history = len(df) >= 200
        df['SMA200'] = df['Close'].rolling(200).mean() if has_macro_history else df['Close'].mean()
        df['SMA50'] = df['Close'].rolling(50).mean() if len(df) >= 50 else df['Close'].mean()
       
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
        df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
       
        curr = df.iloc[-1]
        price, rsi, sma50, sma200, atr = curr['Close'], curr['RSI'], curr['SMA50'], curr['SMA200'], curr['ATR']
       
        historical_vol = df['Volume'].tail(20).mean()
        rvol = curr['Volume'] / historical_vol if historical_vol > 0 else 1.0
       
        dist_from_sma50 = (price / sma50) - 1 if sma50 > 0 else 0.0
        suggested_entry = df['High'].tail(5).max()
        suggested_stop = price - (atr * 2.0)
       
        # --- SCORING MATRIX ALLOCATION ---
        score = 0
        is_above_sma200 = price > sma200 if has_macro_history else True
       
        if is_above_sma200:
            score += 3  

        if 60 <= rsi <= 78:
            score += 4  
        elif rsi > 78:
            score -= 2  

        if rvol >= 2.5:
            score += 3  
        elif rvol >= 1.5:
            score += 2  

        if dist_from_sma50 > 0.35:    
            score -= 4  
           
        # --- ACTION ROUTING SYSTEM ---
        status = "🟡 MONITOR"
        reason = "Awaiting Momentum Confirmation"
       
        if score >= 6 and rvol >= 1.5 and dist_from_sma50 < 0.30 and is_above_sma200:
            status = "🔥 BUY"
            reason = "Strict Institutional Breakout"
        elif enable_analyst_picks and rvol >= 2.0 and 62 <= rsi <= 82 and dist_from_sma50 < 0.40:
            status = "🚀 ANALYST BUY"
            reason = "High Velocity Premarket / Intraday Gap" if rvol >= 3.0 else "Aggressive Momentum Pivot"
        elif price <= suggested_stop:
            status = "🛑 STOP"
            reason = "Volatility Stop Hit"

        # --- NEW: INVESTMENT HORIZON CLASSIFICATION MATRIX ---
        horizon = "N/A"
        if "BUY" in status:
            # 1. Long-Term Hold (Golden Cross or High-Conviction Macro Uptrend)
            if has_macro_history and sma50 > sma200 and price > sma200:
                horizon = "💎 LONG-TERM HOLD"
            # 2. Short-Term Hold (Velocity Rebound or Mean-Reversion Play below macro trendlines)
            else:
                horizon = "⚡ SHORT-TERM SWING"
        elif status == "🛑 STOP":
            horizon = "❌ EXIT POSITION"
        else:
            horizon = "⏳ WATCHLIST"

        # --- BINARY EVENT/EARNINGS SHIELD ---
        earnings_date_str = "N/A"
        try:
            cal = ticker_obj.get_calendar() if hasattr(ticker_obj, 'get_calendar') else None
            if cal is not None and 'Earnings Date' in cal:
                next_earnings = cal['Earnings Date']
                earnings_date_str = next_earnings.strftime('%Y-%m-%d')
                now = datetime.now(timezone.utc) if next_earnings.tzinfo else datetime.now()
                days_to_earnings = (next_earnings - now).days
               
                if 0 <= days_to_earnings <= 5 and "BUY" in status:
                    status = "🟡 HOLD (EARNINGS RISK)"
                    reason = "Immediate Binary Risk Window"
                    horizon = "⚡ RISK REDUCTION"
        except:
            pass

        return {
            "Ticker": symbol,
            "Price": round(price, 2),
            "Score": f"{score}/10",
            "Action": status,
            "Horizon Allocation": horizon,  # Newly injected column field
            "Trigger Reason": reason,
            "Next Earnings": earnings_date_str,
            "Ext%": f"{dist_from_sma50*100:.1f}%",
            "RSI": int(rsi),
            "RVOL": f"{rvol:.1f}x",
            "Entry": round(suggested_entry, 2),
            "Stop": round(suggested_stop, 2),
            "Sizing": f"{int((funds * risk)/(price - suggested_stop)) if (price-suggested_stop)>0 else 0} Shrs"
        }
    except Exception as e:
        return None
        

# --- 5. DATA & UI ---
if check_password():
    st.title("🐋 Institutional Terminal v5.3")
   
    with st.sidebar:
        st.header("⚙️ Core Controls")
        funds = st.number_input("Portfolio Target $", value=100000)
        risk = st.slider("Risk Tolerance %", 0.5, 3.0, 1.5) / 100
       
        st.write("---")
        st.header("🔍 Filter Feeds")
        enable_analyst_picks = st.checkbox("Enable Analyst Overlay Picks", value=True,
                                            help="Allows high-growth velocity stocks to print signals even if below long term 200 SMA lines.")
       
        mode = st.radio("Primary Ticker Feed", ["Watchlist", "Premarket Gainers / Hot Picks 🔥"])
       
        premarket_analyst_picks = "MRAM,ASTS,HIMS,MU,TSLA,QUBT,BZFD"
        user_input = st.text_area("Watchlist Tickers", "NVDA,AMD,HUT,SMCI,ARM")
       
        if mode == "Watchlist":
            t_list = [t.strip().upper() for t in user_input.split(",") if t.strip()]
        else:
            t_list = list(set(get_hot_picks() + [t.strip().upper() for t in premarket_analyst_picks.split(",")]))
           
        run = st.button("🚀 RUN VELOCITY SCAN")

    if run or "results" not in st.session_state:
        res_list = []
        clean_ticker_data = {}
       
        # FIXED: Replacing the broken bulk multi-index download download loop with a robust, isolated iterative downloader pipeline
        with st.spinner("Downloading and processing real-time stock arrays..."):
            for t in t_list:
                try:
                    ticker_obj = yf.Ticker(t)
                    # Fetch data cleanly without multi-index nesting layouts
                    ticker_data = ticker_obj.history(period="2y")
                   
                    if ticker_data is not None and not ticker_data.empty:
                        # Clean column indexing explicitly
                        if isinstance(ticker_data.columns, pd.MultiIndex):
                            ticker_data.columns = ticker_data.columns.get_level_values(0)
                       
                        clean_ticker_data[t] = ticker_data
                        res = analyze_stock(t, ticker_data, ticker_obj, funds, risk, enable_analyst_picks)
                        if res:
                            res_list.append(res)
                except Exception as e:
                    pass # Isolated failures cannot take down the remaining ticker processing queue
               
        st.session_state.results = pd.DataFrame(res_list) if res_list else pd.DataFrame(columns=["Ticker", "Price", "Score", "Action"])
        st.session_state.bulk_data = clean_ticker_data

    tab1, tab2 = st.tabs(["📋 Execution Dashboard", "📈 Technical Visualizer Canvas"])
   
    with tab1:
        st.subheader("Market Sweep Tracking Matrix")
        st.dataframe(st.session_state.results, use_container_width=True, hide_index=True)

    with tab2:
        valid_selections = [t for t in t_list if t in st.session_state.bulk_data]
        if valid_selections:
            sel = st.radio("Asset Pivot View:", valid_selections, horizontal=True)
            if sel and sel in st.session_state.bulk_data:
                df_plot = st.session_state.bulk_data[sel].copy()
               
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
                fig.add_trace(go.Candlestick(x=df_plot.index, open=df_plot['Open'], high=df_plot['High'], low=df_plot['Low'], close=df_plot['Close'], name="Price"), row=1, col=1)
               
                if 'SMA200' in df_plot.columns:
                    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['SMA200'], line=dict(color='gold', width=2), name='SMA 200'), row=1, col=1)
                if 'SMA50' in df_plot.columns:
                    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['SMA50'], line=dict(color='cyan', width=1), name='SMA 50'), row=1, col=1)
               
                fig.add_trace(go.Bar(x=df_plot.index, y=df_plot['Volume'], name='Volume', marker_color='orange'), row=2, col=1)
                fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=650, margin=dict(t=20, b=20, l=20, r=20))
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Awaiting tracking execution matrix loops.")
