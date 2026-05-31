import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# -----------------------------
# App config
# -----------------------------
st.set_page_config(
    page_title="Wealth Terminal v12.0",
    page_icon="💹",
    layout="wide",
)

# -----------------------------
# Simple password gate
# -----------------------------
APP_PASSWORD = "wealth2026"  # change as you like

def check_password():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    pwd = st.text_input("Enter access password", type="password")
    if st.button("Unlock"):
        if pwd == APP_PASSWORD:
            st.session_state.authenticated = True
            st.experimental_rerun()
        else:
            st.error("Incorrect password.")
    return False

if not check_password():
    st.stop()

# -----------------------------
# Strict‑mode AI engine
# -----------------------------
def build_ai_stock_selection_table(historical_data, universe, fundamental_cache):
    """
    Strict mode:
    - Only uses tickers that:
      * exist in historical_data
      * have >= 200 daily bars
      * exist in fundamental_cache
    - If nothing qualifies, returns an empty but well-formed DataFrame.
    """

    base_cols = [
        "Ticker", "Name", "Score", "Return_3M", "Vol_3M",
        "MarketCap", "PE", "Sector", "Country"
    ]

    if historical_data is None or len(historical_data) == 0:
        return pd.DataFrame(columns=base_cols)

    if universe is None or len(universe) == 0:
        return pd.DataFrame(columns=base_cols)

    if isinstance(historical_data.columns, pd.MultiIndex):
        available_tickers = set(historical_data.columns.get_level_values(0))
    else:
        available_tickers = set(historical_data.columns)

    rows = []

    for ticker in universe:
        if ticker not in available_tickers:
            continue
        if ticker not in fundamental_cache:
            continue

        try:
            # Close series
            if isinstance(historical_data.columns, pd.MultiIndex):
                close = historical_data[(ticker, "Close")].dropna()
            else:
                close = historical_data[ticker].dropna()

            if len(close) < 200:
                continue

            window = 63  # ~3 months
            recent = close.tail(window)
            if len(recent) < window:
                continue

            ret_3m = recent.iloc[-1] / recent.iloc[0] - 1.0
            vol_3m = recent.pct_change().dropna().std()

            f = fundamental_cache[ticker]
            mcap = f.get("Market Cap", np.nan)
            pe = f.get("PE", np.nan)
            name = f.get("Name", ticker)
            sector = f.get("Sector", "Unknown")
            country = f.get("Country", "Unknown")

            score = (
                0.6 * ret_3m -
                0.3 * (vol_3m if np.isfinite(vol_3m) else 0) +
                0.1 * (np.log(mcap) if (isinstance(mcap, (int, float)) and mcap > 0) else 0)
            )

            rows.append({
                "Ticker": ticker,
                "Name": name,
                "Score": score,
                "Return_3M": ret_3m,
                "Vol_3M": vol_3m,
                "MarketCap": mcap,
                "PE": pe,
                "Sector": sector,
                "Country": country,
            })

        except Exception:
            # strict but safe: skip any problematic ticker
            continue

    if not rows:
        return pd.DataFrame(columns=base_cols)

    df = pd.DataFrame(rows)
    df = df.sort_values("Score", ascending=False).reset_index(drop=True)
    return df

# -----------------------------
# Data helpers (simple versions)
# -----------------------------
@st.cache_data(show_spinner=False)
def download_history(tickers, start, end):
    if not tickers:
        return pd.DataFrame()
    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
    )
    return data

@st.cache_data(show_spinner=False)
def build_dummy_fundamental_cache(tickers):
    # Placeholder – replace with your real fundamentals source
    cache = {}
    for t in tickers:
        cache[t] = {
            "Name": t,
            "Market Cap": np.random.randint(1e8, 5e11),
            "PE": round(np.random.uniform(5, 40), 1),
            "Sector": "Unknown",
            "Country": "US",
        }
    return cache

# -----------------------------
# Layout
# -----------------------------
st.markdown(
    """
    <style>
    .big-title {
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .sub-title {
        font-size: 14px;
        opacity: 0.7;
        margin-bottom: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="big-title">Wealth Terminal v12.0</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Strict‑mode AI opportunity engine · Experimental build</div>', unsafe_allow_html=True)

tabs = st.tabs(["📊 Overview", "📈 Charts", "🧠 AI Opportunities", "⚙️ Settings"])

# -----------------------------
# Tab 1 – Overview
# -----------------------------
with tabs[0]:
    st.subheader("Market overview")
    st.write("Add your overview KPIs, heatmaps, or summary widgets here.")

# -----------------------------
# Tab 2 – Charts
# -----------------------------
with tabs[1]:
    st.subheader("Price & trend charts")

    default_universe = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    ticker = st.selectbox("Select ticker", default_universe)
    period = st.selectbox("Period", ["6mo", "1y", "2y"], index=1)

    data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if data.empty:
        st.warning("No data for selected ticker.")
    else:
        st.line_chart(data["Close"])

# -----------------------------
# Tab 3 – AI Opportunities (Strict Mode)
# -----------------------------
with tabs[2]:
    st.subheader("AI‑ranked opportunity set (Strict Mode)")

    universe_input = st.text_area(
        "Universe (comma‑separated tickers)",
        value="AAPL, MSFT, GOOGL, AMZN, META, NVDA",
        height=80,
    )
    universe = [t.strip().upper() for t in universe_input.split(",") if t.strip()]

    col1, col2 = st.columns(2)
    with col1:
        years_back = st.slider("History window (years)", 1, 5, 2)
    with col2:
        min_score = st.slider("Minimum score filter", -1.0, 2.0, 0.0, 0.1)

    end = datetime.today()
    start = end - timedelta(days=365 * years_back)

    if st.button("Run strict AI screening"):
        with st.spinner("Synthesizing AI‑ranked opportunity set..."):
            historical_data = download_history(universe, start, end)
            fundamental_cache = build_dummy_fundamental_cache(universe)
            ai_df = build_ai_stock_selection_table(historical_data, universe, fundamental_cache)

        if ai_df is None or ai_df.empty:
            st.warning("No tickers passed strict AI screening (data length, fundamentals, and availability).")
        else:
            filtered = ai_df[ai_df["Score"] >= min_score].reset_index(drop=True)
            st.caption(f"{len(filtered)} tickers passed score ≥ {min_score:.2f} (out of {len(ai_df)} strict‑eligible).")
            st.dataframe(filtered, use_container_width=True)

# -----------------------------
# Tab 4 – Settings / Debug
# -----------------------------
with tabs[3]:
    st.subheader("Settings & debug")

    st.write("Use this tab to tweak parameters, inspect raw data, or debug universe issues.")

    if st.checkbox("Show last downloaded historical_data sample"):
        st.write("Note: this will only show something after you run the AI screening in Tab 3.")
        st.info("In a full app, you’d store the last historical_data in session_state for inspection.")
