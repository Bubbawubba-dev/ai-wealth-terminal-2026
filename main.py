import streamlit as st
import yfinance as yf
import pandas as pd
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timezone

# --- 1. CONFIG ---
st.set_page_config(page_title="Wealth Terminal v5.0", layout="wide")

# --- 2. SCRAPER ---
def get_hot_picks():
    try:
        url = "yahoo.com"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        df_list = pd.read_html(response.text)
        if df_list:
            df = df_list[0] # Fixed: explicitly extracting the concrete dataframe from the list
            if 'Symbol' in df.columns:
                return df['Symbol'].dropna().tolist()[:15]
    except Exception as e:
        pass
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

# --- 4. ANALYTICS ENGINE (HIGH-GROWTH MOMENTUM TECHNICAL VERSION) ---
def analyze_stock(symbol, df, ticker_obj, funds, risk):
    try:
        if df is None or len(df) < 200:
            return None
       
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            return None

        # --- 1. THE MATHEMATICAL MOMENTUM ENGINE ---
        df['SMA200'] = df['Close'].rolling(200).mean()
        df['SMA50'] = df['Close'].rolling(50).mean()
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
        df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
       
        curr = df.iloc[-1]
        price, rsi, sma50, sma200, atr = curr['Close'], curr['RSI'], curr['SMA50'], curr['SMA200'], curr['ATR']
       
        historical_vol = df['Volume'].tail(20).mean()
        rvol = curr['Volume'] / historical_vol if historical_vol > 0 else 1.0
       
        dist_from_sma50 = (price / sma50) - 1
        suggested_entry = df['High'].tail(5).max()
        suggested_stop = price - (atr * 2.0) # Tight institutional stop
       
        # --- 2. SCORING MATRIX ALLOCATION ---
        score = 0
       
        # Macro Trend Alignment (Long-Only Macro Filter)
        if price > sma200:
            score += 3  
        else:
            return None # Rejects micro-cap down-trends immediately

        # High Growth Momentum Sweet Spot (The Power Zone)
        if 60 <= rsi <= 78:
            score += 4  # Strong upward velocity
        elif rsi > 78:
            score -= 3  # Deduct points for immediate short-term extension risk

        # Institutional Volume Inflows
        if rvol >= 2.5:
            score += 3  # High volume breakout surge
        elif rvol >= 1.5:
            score += 2  # Active accumulation sweep

        # Control Extensions
        if dist_from_sma50 > 0.35:    
            score -= 4  
           
        # --- 3. ACTION ROUTING SYSTEM ---
        status = "🟡 MONITOR"
       
        # Tightened execution rules: Requires a score of 6+ to bypass gate restrictions
        if score >= 6 and rvol >= 1.5 and dist_from_sma50 < 0.30:
            status = "🔥 BUY"
        elif price <= suggested_stop:
            status = "🛑 STOP"

        # --- 4. BINARY EVENT/EARNINGS SHIELD ---
        earnings_date_str = "N/A"
        try:
            cal = ticker_obj.get_calendar() if hasattr(ticker_obj, 'get_calendar') else None
            if cal is not None and 'Earnings Date' in cal:
                next_earnings = cal['Earnings Date']
                earnings_date_str = next_earnings.strftime('%Y-%m-%d')
               
                now = datetime.now(timezone.utc) if next_earnings.tzinfo else datetime.now()
                days_to_earnings = (next_earnings - now).days
               
                if 0 <= days_to_earnings <= 5:
                    if status == "🔥 BUY":
                        status = "🟡 HOLD (EARNINGS RISK)"
                    else:
                        status = f"⚠️ EARNINGS ({status})"
        except:
            pass

        return {
            "Ticker": symbol, "Price": round(price, 2), "Score": f"{score}/10",
            "Action": status, "Next Earnings": earnings_date_str,
            "Ext%": f"{dist_from_sma50*100:.1f}%", "RSI": int(rsi), "RVOL": f"{rvol:.1f}x",
            "Entry": round(suggested_entry, 2), "Stop": round(suggested_stop, 2),
            "Sizing": f"{int((funds * risk)/(price - suggested_stop)) if (price-suggested_stop)>0 else 0} Shrs"
        }
    except Exception as e:
        return None
        
        # --- 5. BINARY EVENT/EARNINGS SHIELD ---
        earnings_date_str = "N/A"
        try:
            cal = ticker_obj.get_calendar() if hasattr(ticker_obj, 'get_calendar') else None
            if cal is not None and 'Earnings Date' in cal:
                next_earnings = cal['Earnings Date']
                earnings_date_str = next_earnings.strftime('%Y-%m-%d')
               
                now = datetime.now(timezone.utc) if next_earnings.tzinfo else datetime.now()
                days_to_earnings = (next_earnings - now).days
               
                # Risk mitigation rule: Modify actions to prevent buying directly ahead of earnings reports
                if 0 <= days_to_earnings <= 5:
                    if status == "🔥 BUY":
                        status = "🟡 HOLD (EARNINGS RISK)"
                    else:
                        status = f"⚠️ EARNINGS ({status})"
        except:
            pass

        return {
            "Ticker": symbol, "Price": round(price, 2), "Score": f"{score}/10",
            "Action": status, "Rev Growth": f"{rev_growth*100:.1f}%", "ROE": f"{roe*100:.1f}%",
            "Ext%": f"{dist_from_sma50*100:.1f}%", "RSI": int(rsi), "RVOL": f"{rvol:.1f}x",
            "Entry": round(suggested_entry, 2), "Stop": round(suggested_stop, 2),
            "Sizing": f"{int((funds * risk)/(price - suggested_stop)) if (price-suggested_stop)>0 else 0} Shrs"
        }
    except Exception as e:
        return None
        
# --- 5. DATA & UI ---
if check_password():
    st.title("🐋 Institutional Terminal v5.0")
    with st.sidebar:
        funds = st.number_input("Portfolio $", value=100000)
        risk = st.slider("Risk %", 0.5, 3.0, 1.5) / 100
        mode = st.radio("Scanner Mode", ["Watchlist", "Hot Picks 🔥"])
       
        default_tickers = "NVDA,AMD,HUT,SMCI,ARM"
        user_input = st.text_area("Tickers", default_tickers)
       
        if mode == "Watchlist":
            t_list = [t.strip().upper() for t in user_input.split(",") if t.strip()]
        else:
            t_list = get_hot_picks()
           
        run = st.button("🚀 EXECUTE SCAN")

    if run or "results" not in st.session_state:
        # Crucial Fix: Explicitly pass group_by='ticker' AND flatten the resulting multi-index
        bulk_df = yf.download(t_list, period="2y", group_by='ticker', progress=False)
       
        res_list = []
        clean_ticker_data = {}
       
        for t in t_list:
            try:
                # Isolate target asset slice safely from yfinance MultiIndex output dataframe
                if len(t_list) > 1:
                    if t in bulk_df.columns.levels[0]:
                        ticker_data = bulk_df[t].copy()
                    else:
                        continue
                else:
                    ticker_data = bulk_df.copy()
                    # If single-ticker download skips MultiIndex generation, normalize explicitly
                    if isinstance(ticker_data.columns, pd.MultiIndex):
                        ticker_data.columns = ticker_data.columns.get_level_values(0)
               
                # Strip any remnant multi-level names left over by the yfinance engine
                if isinstance(ticker_data.columns, pd.MultiIndex):
                    ticker_data.columns = ticker_data.columns.get_level_values(0)
               
                clean_ticker_data[t] = ticker_data
                ticker_obj = yf.Ticker(t)
               
                res = analyze_stock(t, ticker_data, ticker_obj, funds, risk)
                if res:
                    res_list.append(res)
            except Exception as e:
                pass
               
        st.session_state.results = pd.DataFrame(res_list) if res_list else pd.DataFrame(columns=["Ticker", "Price", "Score", "Action"])
        st.session_state.bulk_data = clean_ticker_data

    tab1, tab2 = st.tabs(["📋 Execution Dashboard", "📈 Indicator Analysis Canvas"])
   
    with tab1:
        st.dataframe(st.session_state.results, use_container_width=True, hide_index=True)

    with tab2:
        # Dynamically protect tab rendering against un-scanned assets or processing omissions
        valid_selections = [t for t in t_list if t in st.session_state.bulk_data]
        if valid_selections:
            sel = st.radio("Asset Pivot View:", valid_selections, horizontal=True)
            if sel and sel in st.session_state.bulk_data:
                df_plot = st.session_state.bulk_data[sel].copy()
               
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
                fig.add_trace(go.Candlestick(x=df_plot.index, open=df_plot['Open'], high=df_plot['High'], low=df_plot['Low'], close=df_plot['Close'], name="Price"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['Close'].rolling(200).mean(), line=dict(color='gold', width=2), name='SMA 200'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['Close'].rolling(50).mean(), line=dict(color='cyan', width=1), name='SMA 50'), row=1, col=1)
               
                # Bottom panel: Volume & Technical confirmation
                fig.add_trace(go.Bar(x=df_plot.index, y=df_plot['Volume'], name='Volume', marker_color='orange'), row=2, col=1)
               
                fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=650, margin=dict(t=20, b=20, l=20, r=20))
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No single-level historical data matrix available to build technical visualizer sweeps.")
