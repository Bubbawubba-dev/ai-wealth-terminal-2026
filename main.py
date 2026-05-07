import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import requests
import time

# --- 🛡️ SECURITY ---
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

# --- ⚙️ BULK DATA ENGINE ---
@st.cache_data(ttl=600)
def get_bulk_data(tickers):
    # Download all price data at once (Faster than one-by-one)
    df_all = yf.download(tickers, period="1y", group_by='ticker', progress=False)
    return df_all

# --- 🧠 LOGIC ENGINE ---
def analyze_stock(symbol, df, funds, risk):
    try:
        if df.empty or len(df) < 200:
            return None
           
        # Indicators
        df['SMA200'] = df['Close'].rolling(window=200).mean()
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / loss)))
        df['ATR'] = (df['High'] - df['Low']).rolling(window=14).mean()

        curr = df.iloc[-1]
        price, rsi, sma200, atr = curr['Close'], curr['RSI'], curr['SMA200'], curr['ATR']
        rvol = curr['Volume'] / df['Volume'].tail(20).mean()

        # --- 🛡️ ANTI-FAKEOUT LOGIC ---
        score = 0
        status = "🟡 HOLD"
       
        # 1. Trend Strength (Regime)
        if price > sma200: score += 3
       
        # 2. Volume Conviction (The Fakeout Killer)
        if rvol > 2.0: score += 4
        elif rvol > 1.2: score += 2
       
        # 3. Momentum Room (RSI)
        if 50 < rsi < 68: score += 3  # Sweet spot for new breakouts
        elif rsi > 75: score -= 2     # Deduct points for being overextended

        # --- ACTION TRIGGERS ---
        if score >= 8 and rvol > 2.0:
            status = "🔥 STRONG BREAKOUT"
        elif score >= 6 and rsi < 35:
            status = "💎 BUY DIP"
        elif rsi > 80:
            status = "🛑 TAKE PROFIT"
        elif rvol < 1.0 and price > df['High'].shift(1).iloc[-1]:
            status = "⚠️ FAKEOUT RISK" # Price up, but no one is buying

        stop_dist = (atr * 2)
        shares = int((funds * risk) / stop_dist) if stop_dist > 0 else 0

        return {
            "Ticker": symbol,
            "Price": f"${price:.2f}",
            "Score": f"{score}/10",
            "RVOL": f"{rvol:.1f}x",
            "RSI": round(rsi, 1),
            "Action": status,
            "Sizing": f"{shares} Shrs"
        }
    except:
        return None

# --- 🖥️ UI ---
if check_password():
  # --- GLOBAL STYLING ---
st.markdown("""
<style>
/* Background gradient */
.main {
    background: linear-gradient(135deg, #0f0f0f 0%, #1a1f2b 100%);
}

/* Glassmorphism cards */
.card {
    padding: 20px;
    border-radius: 15px;
    background: rgba(255,255,255,0.06);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.15);
    margin-bottom: 20px;
}

/* Neon KPI text */
.big-kpi {
    font-size: 40px;
    font-weight: 700;
    color: #00eaff;
    text-shadow: 0 0 12px #00eaff;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(10px);
    border-right: 1px solid rgba(255,255,255,0.1);
}
</style>
""", unsafe_allow_html=True)


# --- 🖥️ UI ---
if check_password():
    st.set_page_config(page_title="Wealth Terminal", layout="wide")
    st.title("🐋 Institutional Wealth Terminal 2026")

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("🕹️ Strategy Parameters")
        funds = st.number_input("Balance ($)", value=100000)
        risk = st.slider("Risk (%)", 0.5, 3.0, 1.5) / 100

        mode = st.selectbox("Market Feed", ["Core Watchlist", "Pre-Market Hot Picks"])
        if mode == "Core Watchlist":
            user_list = st.text_area("Tickers (Comma Separated)", "NVDA,AAPL,MSFT,TSLA,AMD,HUT,SMCI,AVGO")
        else:
            user_list = "HUT,AMD,SMCI,COMP,VEEV,PLTR,MARA,RIOT,COIN,HOOD,MSTR,SOXL"

        refresh = st.button("♻️ Run Scanner")

    tickers = [t.strip().upper() for t in user_list.split(",") if t]

    # --- SCANNER ---
    if refresh or "results" not in st.session_state:
        with st.spinner(f"Scanning {len(tickers)} assets..."):
            bulk_df = get_bulk_data(tickers)
            results = []
            for t in tickers:
                df = bulk_df[t] if len(tickers) > 1 else bulk_df
                res = analyze_stock(t, df, funds, risk)
                if res:
                    results.append(res)

            st.session_state.results = pd.DataFrame(results)

    # --- NAVIGATION TABS ---
    tab_overview, tab_risk, tab_execution, tab_ai = st.tabs(
        ["📊 Overview", "🔥 Risk", "📋 Execution", "🤖 AI Insights"]
    )


    # ---------------------------
    # 📊 OVERVIEW TAB
    # ---------------------------
    with tab_overview:

        # --- MARKET RADAR (D) ---
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("📡 Market Radar")

        df = st.session_state.results.copy()
        df['AbsReturn'] = df['Return'].abs()
        df = df.sort_values('AbsReturn', ascending=False)

        cols = st.columns(6)

        for i, (_, row) in enumerate(df.head(6).iterrows()):
            color = "#00ff99" if row['Return'] > 0 else "#ff4d4d"
            arrow = "▲" if row['Return'] > 0 else "▼"

            cols[i].markdown(
                f"""
                <div style='padding:14px; border-radius:12px; 
                background:rgba(255,255,255,0.06); 
                border:1px solid rgba(255,255,255,0.15); 
                backdrop-filter: blur(10px);
                text-align:center;'>
                    <div style='font-size:22px; font-weight:700;'>{row['Ticker']}</div>
                    <div style='font-size:20px; color:{color}; font-weight:600;'>{arrow} {row['Return']:.2f}%</div>
                    <div style='opacity:0.7; font-size:14px;'>Vol: {row['Volatility']:.2f}%</div>
                    <div style='opacity:0.7; font-size:14px;'>Sharpe: {row['Sharpe']:.2f}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("</div>", unsafe_allow_html=True)


        # --- KPIs ---
        if "results" in st.session_state:
            df = st.session_state.results

            colA, colB, colC = st.columns(3)
            colA.markdown(f"<div class='big-kpi'>{df['Return'].mean():.2f}%</div>Avg Return", unsafe_allow_html=True)
            colB.markdown(f"<div class='big-kpi'>{df['Volatility'].mean():.2f}%</div>Volatility", unsafe_allow_html=True)
            colC.markdown(f"<div class='big-kpi'>{df['Sharpe'].mean():.2f}</div>Sharpe", unsafe_allow_html=True)

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)


    # ---------------------------
    # 🔥 RISK TAB
    # ---------------------------
    with tab_risk:
        st.subheader("🔥 Risk Correlation Heatmap")

        if "corr" in st.session_state:
            corr_df = st.session_state.corr

            fig = px.imshow(
                corr_df,
                color_continuous_scale="RdYlGn",
                aspect="auto",
                title="Risk Correlation Matrix",
            )
            fig.update_layout(
                template="plotly_dark",
                margin=dict(l=20, r=20, t=40, b=20),
                coloraxis_colorbar=dict(title="Correlation")
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.dataframe(
                corr_df.style.background_gradient(cmap='RdYlGn'),
                use_container_width=True
            )
            st.markdown("</div>", unsafe_allow_html=True)


    # ---------------------------
    # 📋 EXECUTION TAB
    # ---------------------------
    with tab_execution:
        st.subheader("📋 Market Execution")

        if "results" in st.session_state:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.dataframe(
                st.session_state.results,
                use_container_width=True,
                hide_index=True
            )
            st.markdown("</div>", unsafe_allow_html=True)


    # ---------------------------
    # 🤖 AI INSIGHTS TAB
    # ---------------------------
    with tab_ai:
        st.subheader("🤖 AI Insights")

        if "results" in st.session_state:
            df = st.session_state.results

            # --- AI TRADE IDEAS (C) ---
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.write("### 💡 AI‑Generated Trade Ideas")

            top_return = df.iloc[df['Return'].idxmax()]
            top_sharpe = df.iloc[df['Sharpe'].idxmax()]
            low_risk = df.iloc[df['Volatility'].idxmin()]
            high_vol = df.iloc[df['Volatility'].idxmax()]

            st.write(f"**1️⃣ Momentum Long:** `{top_return['Ticker']}` — strongest short‑term return profile.")
            st.write(f"**2️⃣ Quality Long:** `{top_sharpe['Ticker']}` — best risk‑adjusted performance.")
            st.write(f"**3️⃣ Defensive Pick:** `{low_risk['Ticker']}` — lowest volatility in the basket.")
            st.write(f"**4️⃣ Pair Trade:** Long {top_sharpe['Ticker']} / Short {high_vol['Ticker']}")
            basket = ", ".join(df.nlargest(3, 'Sharpe')['Ticker'].tolist())
            st.write(f"**5️⃣ Smart‑Beta Basket:** {basket}")

            st.markdown("</div>", unsafe_allow_html=True)
        st.dataframe(st.session_state.corr.style.background_gradient(cmap='RdYlGn'), use_container_width=True)
