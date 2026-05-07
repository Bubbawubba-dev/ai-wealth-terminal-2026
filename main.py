import streamlit as st
import pandas as pd
import plotly.express as px

# ---------------------------
# YOUR EXISTING LOGIC GOES HERE
# ---------------------------
# def check_password():
#     ...
#
# def get_bulk_data(tickers):
#     ...
#
# def analyze_stock(ticker, df, funds, risk):
#     ...
# ---------------------------

# Must be the first Streamlit call
st.set_page_config(page_title="Wealth Terminal", layout="wide")

# --- GLOBAL STYLING ---
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #0f0f0f 0%, #1a1f2b 100%);
}
.card {
    padding: 20px;
    border-radius: 15px;
    background: rgba(255,255,255,0.06);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.15);
    margin-bottom: 20px;
}
.big-kpi {
    font-size: 40px;
    font-weight: 700;
    color: #00eaff;
    text-shadow: 0 0 12px #00eaff;
}
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(10px);
    border-right: 1px solid rgba(255,255,255,0.1);
}
</style>
""", unsafe_allow_html=True)


# --- 🖥️ UI ---
if check_password():

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

    # --- TABS ---
    tab_overview, tab_risk, tab_execution, tab_ai = st.tabs(
        ["📊 Overview", "🔥 Risk", "📋 Execution", "🤖 AI Insights"]
    )

    # ---------------------------
    # 📊 OVERVIEW TAB
    # ---------------------------
    with tab_overview:

        if "results" in st.session_state:
            # MARKET RADAR
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("📡 Market Radar")

            df = st.session_state.results.copy()
            df["AbsReturn"] = df["Return"].abs()
            df = df.sort_values("AbsReturn", ascending=False)

            cols = st.columns(6)
            for i, (_, row) in enumerate(df.head(6).iterrows()):
                color = "#00ff99" if row["Return"] > 0 else "#ff4d4d"
                arrow = "▲" if row["Return"] > 0 else "▼"
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

            # KPIs + table
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
                corr_df.style.background_gradient(cmap="RdYlGn"),
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

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.write("### 💡 AI‑Generated Trade Ideas")

            top_return = df.iloc[df["Return"].idxmax()]
            top_sharpe = df.iloc[df["Sharpe"].idxmax()]
            low_risk = df.iloc[df["Volatility"].idxmin()]
            high_vol = df.iloc[df["Volatility"].idxmax()]

            st.write(f"**1️⃣ Momentum Long:** `{top_return['Ticker']}` — strongest short‑term return.")
            st.write(f"**2️⃣ Quality Long:** `{top_sharpe['Ticker']}` — best risk‑adjusted performance.")
            st.write(f"**3️⃣ Defensive Pick:** `{low_risk['Ticker']}` — lowest volatility.")
            st.write(f"**4️⃣ Pair Trade:** Long {top_sharpe['Ticker']} / Short {high_vol['Ticker']}")
            basket = ", ".join(df.nlargest(3, "Sharpe")["Ticker"].tolist())
            st.write(f"**5️⃣ Smart‑Beta Basket:** {basket}")

            st.markdown("</div>", unsafe_allow_html=True)
