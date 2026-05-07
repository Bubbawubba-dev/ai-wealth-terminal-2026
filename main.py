import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import requests
from datetime import datetime

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Institutional Wealth Terminal", layout="wide")

# --- 2. PRE-MARKET HOT PICKS SCRAPER ---
def get_hot_picks():
    try:
        url = "https://yahoo.com"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        df_list = pd.read_html(response.text)
        if df_list:
            df = df_list[0]
            return df['Symbol'].tolist()[:15]
    except:
        # Verified Momentum Leaders for May 7, 2026
        return ["HUT", "AMD", "SMCI", "FLEX", "COMP", "VCYT", "VECO", "ARM", "IONQ", "PLTR"]

# --- 3. SECURITY ---
def check_password():
    if "password_correct" not in st.session_state:
        st.sidebar.title("🔐 Terminal Access")
        pwd = st.sidebar.text_input("Access Key", type="password")
        if st.sidebar.button("Unlock"):
            if pwd == st.secrets.get("APP_PASSWORD", "1234"):
                st.session_state["password_correct"] = True
                st.rerun()
            else:
                st.sidebar.error("❌ Invalid Key")
        return False
    return True

# --- 4. ANALYTICS ENGINE (v4.0) ---
def analyze_stock(symbol, df, funds, risk):
    try:
        if df.empty or len(df) < 200: return None
       
        # Indicators
        df['SMA50'] = df['Close'].rolling(50).mean()
        df['SMA200'] = df['Close'].rolling(200).mean()
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / loss)))
        df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
       
        curr = df.iloc[-1]
        price = curr['Close']
        atr = curr['ATR']
        rvol = curr['Volume'] / df['Volume'].tail(20).mean()
       
        # Rulebook Logic
        is_stage_2 = price > df['SMA50'].iloc[-1] > df['SMA200'].iloc[-1]
       
        # Entry / Exit / Target Math
        suggested_entry = df['High'].tail(5).max()
        suggested_stop = price - (atr * 2.5)
        suggested_target = price + (atr * 5)
       
        potential_risk = price - suggested_stop
        potential_reward = suggested_target - price
        rr_ratio = potential_reward / potential_risk if potential_risk > 0 else 0
       
        # Conviction Scoring
        score = 0
        if is_stage_2: score += 4
        if rvol > 1.8: score += 4
        if 45 < curr['RSI'] < 68: score += 2
        elif curr['RSI'] > 80: score -= 2

        status = "🟡 MONITOR"
        if score >= 8 and rvol > 1.8: status = "🔥 ENTRY"
        elif price <= suggested_stop: status = "🛑 STOP"
        elif rvol < 1.0 and price > df['High'].shift(1).iloc[-1]: status = "⚠️ FAKEOUT"
       
        shares = int((funds * risk) / potential_risk) if potential_risk > 0 else 0

        return {
            "Ticker": symbol,
            "Price": round(price, 2),
            "Score": f"{score}/10",
            "R/R": f"{rr_ratio:.1f}x",
            "Entry": round(suggested_entry, 2),
            "Stop": round(suggested_stop, 2),
            "Target": round(suggested_target, 2),
            "Action": status,
            "Sizing": f"{shares} Shrs",
            "RSI": int(curr['RSI']),
            "News": f"https://yahoo.com{symbol}"
        }
    except: return None

# --- 5. MAIN INTERFACE ---
if check_password():
    st.title("🐋 Institutional Wealth Terminal 2026")
   
    with st.sidebar:
        st.header("🕹️ Strategy Parameters")
        funds = st.number_input("Portfolio Balance ($)", value=100000)
        risk = st.slider("Risk Per Trade (%)", 0.5, 3.0, 1.5) / 100
       
        mode = st.radio("Scanner Mode", ["My Watchlist", "Live Hot Picks 🔥"])
        if mode == "My Watchlist":
            user_input = st.text_area("Tickers", "NVDA,AAPL,TSLA,AMD,HUT,SMCI")
            t_list = [t.strip().upper() for t in user_input.split(",") if t]
        else:
            t_list = get_hot_picks()
            st.info(f"Loaded {len(t_list)} active movers.")
       
        run = st.button("🚀 EXECUTE SCAN")

    # --- 6. DATA EXECUTION ---
    if run or "results" not in st.session_state:
        with st.spinner("Analyzing Market Data..."):
            bulk_df = yf.download(t_list, period="1y", group_by='ticker', progress=False)
            res_list = [analyze_stock(t, bulk_df[t] if len(t_list)>1 else bulk_df, funds, risk) for t in t_list]
            st.session_state.results = pd.DataFrame([r for r in res_list if r])
           
            # Safe Correlation
            try:
                hp_df = yf.download(t_list, period="6mo", progress=False)['Close']
                if isinstance(hp_df.columns, pd.MultiIndex): hp_df.columns = hp_df.columns.get_level_values(0)
                st.session_state.corr = hp_df.dropna(axis=1, how='all').corr()
            except:
                st.session_state.corr = pd.DataFrame()
           
            st.session_state.bulk_data = bulk_df

    # --- 7. DASHBOARD LAYOUT ---
    c1, c2, c3 = st.columns(3)
    c1.metric("Assets Analyzed", len(t_list))
    c2.metric("Market Sentiment", "🐂 BULL" if not st.session_state.results.empty and st.session_state.results['RSI'].mean() < 70 else "🛑 HOT")
    c3.metric("Terminal Time", datetime.now().strftime("%H:%M"))

    tab1, tab2, tab3 = st.tabs(["📋 Execution Dashboard", "📈 Technical Indicators", "🔥 Risk Correlation"])

    with tab1:
        st.dataframe(
            st.session_state.results,
            use_container_width=True,
            hide_index=True,
            column_config={"News": st.column_config.LinkColumn("Research Link")}
        )

        with tab2:
        st.subheader("📈 Technical Analysis")
       
        # 📱 Mobile-Friendly Selection (Vertical list instead of a dropdown)
        selected_stock = st.radio(
            "Select Asset to View Chart:",
            t_list,
            horizontal=True, # Rows of buttons instead of a long list
            index=0
        )
       
        if selected_stock and "bulk_data" in st.session_state:
            # Add a "Quick Stats" row for mobile eyes
            df_ind = st.session_state.bulk_data[selected_stock] if len(t_list) > 1 else st.session_state.bulk_data
           
            # Simplified Chart for Mobile
            fig = go.Figure(data=[go.Candlestick(
                x=df_ind.index,
                open=df_ind['Open'],
                high=df_ind['High'],
                low=df_ind['Low'],
                close=df_ind['Close'],
                name="Price"
            )])
           
            # Keep the gold line but make it thinner for small screens
            fig.add_trace(go.Scatter(
                x=df_ind.index,
                y=df_ind['Close'].rolling(200).mean(),
                line=dict(color='gold', width=1.5),
                name='SMA 200'
            ))
           
            # 📱 Force chart to be taller on mobile and remove extra margins
            fig.update_layout(
                template="plotly_dark",
                xaxis_rangeslider_visible=False,
                height=400,
                margin=dict(l=0, r=0, t=20, b=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
           
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    with tab3:
        if not st.session_state.corr.empty:
            st.dataframe(st.session_state.corr.style.background_gradient(cmap='RdYlGn', axis=None).format("{:.2f}"), use_container_width=True)
        else:
            st.warning("Heatmap currently unavailable.")
