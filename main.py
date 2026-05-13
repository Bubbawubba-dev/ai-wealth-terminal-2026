import streamlit as st
import yfinance as yf
import pandas as pd
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timezone

# --- 1. CONFIG ---
st.set_page_config(page_title="Wealth Terminal v5.7", layout="wide")

# --- 2. SCRAPER ---
@st.cache_data(ttl=3600)
def get_micro_cap_universe():
    try:
        url = "wikipedia.org"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers, timeout=10)
        df_list = pd.read_html(response.text)
        for df in df_list:
            if any('Ticker' in col or 'Symbol' in col for col in df.columns):
                col_name = [col for col in df.columns if 'Ticker' in col or 'Symbol' in col]
                tickers = df[col_name[0]].dropna().astype(str).tolist()
                clean_tickers = []
                for t in tickers:
                    token = t.split(':')[-1].replace(')', '').strip().upper()
                    if token.isalpha() and len(token) <= 5:
                        clean_tickers.append(token)
                if clean_tickers:
                    return list(set(clean_tickers))[:25]
    except:
        pass
    return ["MRAM", "ASTS", "HIMS", "QUBT", "BZFD", "HUT", "FLEX", "VCYT", "VECO", "IONQ"]

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

# --- 4. ANALYTICS ENGINE (SYNCHRONISED xVOL DEFINITION) ---
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
       
        # Calculate Volume Velocity (xVOL)
        short_term_vol = df['Volume'].tail(3).mean()
        historical_base = df['Volume'].tail(60).mean()
        xvol = short_term_vol / historical_base if historical_base > 0 else 1.0
       
        dist_from_sma50 = (price / sma50) - 1 if sma50 > 0 else 0.0
        suggested_entry = df['High'].tail(5).max()
        suggested_stop = price - (atr * 1.8)
       
        # Scoring Matrix
        score = 0
        is_above_sma200 = price > sma200 if has_macro_history else True
        if is_above_sma200: score += 3  

        if 65 <= rsi <= 82: score += 5  
        elif rsi > 82: score -= 2  

        if xvol >= 4.0: score += 4  
        elif xvol >= 2.0: score += 2  

        if dist_from_sma50 > 0.40: score -= 4  
           
        # Action Routing
        status = "🟡 MONITOR"
        reason = "Awaiting Momentum Confirmation"
       
        if score >= 6 and xvol >= 2.0 and dist_from_sma50 < 0.35:
            status = "🔥 BUY"
            reason = "Explosive Volume Breakout Run"
        elif price <= suggested_stop:
            status = "🛑 STOP"
            reason = "Volatility Stop Hit"

        # Horizon Allocation Profile Engine
        horizon = "⏳ WATCHLIST"
        if "BUY" in status:
            horizon = "⚡ SHORT-TERM SWING" if xvol >= 3.0 else "💎 LONG-TERM HOLD"
        elif status == "🛑 STOP":
            horizon = "❌ EXIT POSITION"

        # Earnings Shield
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

        # KEY FIX: Explicitly mapping output key to match the sorting engine name
        return {
            "Ticker": symbol,
            "Price": round(price, 2),
            "Score": f"{score}/10",
            "Action": status,
            "Horizon Allocation": horizon,
            "Trigger Reason": reason,
            "Next Earnings": earnings_date_str,
            "Ext%": f"{dist_from_sma50*100:.1f}%",
            "RSI": int(rsi),
            "xVOL Velocity": f"{xvol:.1f}x",  # Synchronised exact key string match
            "Entry": round(suggested_entry, 2),
            "Stop": round(suggested_stop, 2),
            "Sizing": f"{int((funds * risk)/(price - suggested_stop)) if (price-suggested_stop)>0 else 0} Shrs"
        }
    except:
        return None

# --- 5. DATA & UI ENVIRONMENT (DEFENSIVE SORTING ENGINE) ---
if check_password():
    st.title("🐋 Institutional Micro-Cap Terminal v5.7")
   
    with st.sidebar:
        st.header("⚙️ Capital Allocator")
        funds = st.number_input("Portfolio Target Deployment $", value=100000)
        risk = st.slider("Risk Per Trade Tolerance %", 0.5, 3.0, 1.5) / 100
       
        st.write("---")
        st.header("🔍 Index Feed Filters")
        enable_analyst_picks = st.checkbox("Enable Velocity Overlays", value=True)
        feed_mode = st.radio("Active Engine Feed Source", ["Scrape Automated Micro-Cap Index 🚀", "Manual Watchlist Tickers 📋"])
       
        if "Manual Watchlist Tickers 📋" in feed_mode:
            user_input = st.text_area("Watchlist Input", "MRAM,ASTS,HIMS,QUBT,BZFD")
            t_list = [t.strip().upper() for t in user_input.split(",") if t.strip()]
        else:
            with st.spinner("Scraping index..."):
                t_list = get_micro_cap_universe()
            st.info(f"Scraped Tickers Locked: {', '.join(t_list)}")
           
        st.write("---")
        st.header("📊 Institutional Sorting Engine")
        sort_by = st.selectbox("Rank Breakout Priority By:", ["Technical Score", "Volume Velocity (xVOL)", "Momentum Velocity (RSI)", "Extension Level (Ext%)"])
        sort_order = st.radio("Sort Order Direction:", ["Highest First 📈", "Lowest First 📉"])
        ascending_bool = sort_order == "Lowest First 📉"
       
        run = st.button("🚀 EXECUTE ALPHA VELOCITY SWEEP")

    if run or "results" not in st.session_state:
        res_list = []
        clean_ticker_data = {}
       
        for t in t_list:
            try:
                ticker_obj = yf.Ticker(t)
                ticker_data = ticker_obj.history(period="2y")
                if ticker_data is not None and not ticker_data.empty:
                    if isinstance(ticker_data.columns, pd.MultiIndex):
                        ticker_data.columns = ticker_data.columns.get_level_values(0)
                    clean_ticker_data[t] = ticker_data
                    res = analyze_stock(t, ticker_data, ticker_obj, funds, risk, enable_analyst_picks)
                    if res: res_list.append(res)
            except:
                pass
               
        if res_list:
            raw_df = pd.DataFrame(res_list)
           
            # Defensive Column Verification to catch missing parameters
            if 'xVOL Velocity' in raw_df.columns:
                raw_df['RVOL_num'] = raw_df['xVOL Velocity'].astype(str).str.replace('x', '', regex=False).astype(float)
            else:
                raw_df['RVOL_num'] = 1.0
               
            if 'Ext%' in raw_df.columns:
                raw_df['Ext_num'] = raw_df['Ext%'].astype(str).str.replace('%', '', regex=False).astype(float)
            else:
                raw_df['Ext_num'] = 0.0
               
            if 'Score' in raw_df.columns:
                raw_df['Score_num'] = raw_df['Score'].astype(str).str.split('/').str[0].astype(int)
            else:
                raw_df['Score_num'] = 0
           
            sort_map = {
                "Technical Score": "Score_num",
                "Volume Velocity (xVOL)": "RVOL_num",
                "Momentum Velocity (RSI)": "RSI",
                "Extension Level (Ext%)": "Ext_num"
            }
           
            target_column = sort_map.get(sort_by, "Score_num")
            sorted_df = raw_df.sort_values(by=target_column, ascending=ascending_bool)
            st.session_state.results = sorted_df.drop(columns=['RVOL_num', 'Ext_num', 'Score_num'], errors='ignore')
        else:
            st.session_state.results = pd.DataFrame(columns=["Ticker", "Price", "Score", "Action", "Horizon Allocation"])
           
        st.session_state.bulk_data = clean_ticker_data

    tab1, tab2 = st.tabs(["📋 Execution Dashboard", "📈 Technical Visualizer Canvas"])
    with tab1:
        st.subheader(f"Micro-Cap Momentum Sweep (Sorted by {sort_by})")
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
