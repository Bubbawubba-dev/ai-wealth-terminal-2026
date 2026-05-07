import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
import yfinance as yf

# ---------------------------------------------------------------------------
# CONFIG / SETUP
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Wealth Terminal v3.6", layout="wide")

# Configure basic logging (stdout in Streamlit logs)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Approximate sector forward P/E benchmarks (static; adjust as needed)
SECTOR_FORWARD_PE_BENCH: Dict[str, float] = {
    "Technology": 25.0,
    "Communication Services": 20.0,
    "Consumer Cyclical": 22.0,           # yfinance often uses "Consumer Cyclical"
    "Consumer Discretionary": 22.0,
    "Financial Services": 13.0,
    "Financial": 13.0,
    "Industrials": 18.0,
    "Healthcare": 18.0,
    "Health Care": 18.0,
    "Energy": 10.0,
    "Basic Materials": 15.0,
    "Materials": 15.0,
    "Real Estate": 15.0,
    "Utilities": 14.0,
}
DEFAULT_MARKET_FORWARD_PE: float = 18.0


# ---------------------------------------------------------------------------
# HOT PICKS / WATCHLIST
# ---------------------------------------------------------------------------

def get_hot_picks() -> List[str]:
    """
    Static high-velocity list. In production, this would come from
    a model, feed or database.
    """
    return [
        "HUT", "AMD", "SMCI", "NVDA", "AAPL", "MSFT", "TSLA",
        "PLTR", "MARA", "MSTR", "SOXL", "COIN", "ARM"
    ]


# ---------------------------------------------------------------------------
# SECURITY
# ---------------------------------------------------------------------------

def check_password() -> bool:
    """
    Basic password gate using Streamlit secrets.

    NOTE: This is NOT a full security layer. It only protects casual use.
    For production, add proper auth (SAML/OAuth/Okta, etc.) in front.
    """
    if "password_correct" not in st.session_state:
        st.sidebar.title("🔐 Terminal Access")
        pwd = st.sidebar.text_input("Access Key", type="password")
        if st.sidebar.button("Unlock"):
            expected = st.secrets.get("APP_PASSWORD", None)
            if expected is None:
                st.sidebar.error("APP_PASSWORD not configured in secrets.")
                logger.warning("APP_PASSWORD missing from secrets.")
                return False

            if pwd == expected:
                st.session_state["password_correct"] = True
                st.rerun()
            else:
                st.sidebar.error("❌ Invalid Key")
                logger.info("Incorrect password attempt.")
        return False
    return True


# ---------------------------------------------------------------------------
# ANALYTICS HELPERS
# ---------------------------------------------------------------------------

def get_sector_benchmark(sector: Optional[str]) -> float:
    """
    Returns a static forward P/E benchmark for the given sector.
    If sector is missing or unknown, returns a default market baseline.
    """
    if not sector:
        return DEFAULT_MARKET_FORWARD_PE
    return SECTOR_FORWARD_PE_BENCH.get(sector, DEFAULT_MARKET_FORWARD_PE)


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute RSI for a price series using a simple average method.
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Compute a basic ATR: average of (high - low) over the period.
    This is not Wilder's true range ATR, but adequate for a simple risk band.
    """
    high_low = df["High"] - df["Low"]
    atr = high_low.rolling(period).mean()
    return atr


def fetch_price_history(
    tickers: List[str],
    period: str = "1y"
) -> pd.DataFrame:
    """
    Download OHLCV history for a list of tickers using yfinance.
    Returns a multi-indexed or single-index DataFrame as per yfinance.

    NOTE: yfinance is best-effort; data may be:
          - Delayed by ~15 minutes
          - Adjusted for splits/divs
          - Missing or incorrect for some tickers
    """
    if not tickers:
        return pd.DataFrame()
    try:
        df = yf.download(
            tickers,
            period=period,
            group_by="ticker" if len(tickers) > 1 else "column",
            progress=False,
            auto_adjust=False,       # keep raw OHLC
            threads=True,
        )
        return df
    except Exception as exc:
        logger.exception("Error fetching price history: %s", exc)
        return pd.DataFrame()


def fetch_ticker_info(symbol: str) -> Dict:
    """
    Safely fetch ticker fundamentals via yfinance.

    yfinance uses Yahoo Finance, which:
      - Is not an official data feed
      - May be stale, incomplete or inconsistent
      - Should not be used for regulatory reporting or P&L.

    In production, replace with a licensed data provider (Bloomberg, Refinitiv, etc.).
    """
    try:
        t = yf.Ticker(symbol)
        # Using .get_info() (if available) is often more stable with newer yfinance.
        # If running older yfinance, fallback to .info
        get_info = getattr(t, "get_info", None)
        if callable(get_info):
            return get_info() or {}
        return getattr(t, "info", {}) or {}
    except Exception as exc:
        logger.warning("Failed to fetch info for %s: %s", symbol, exc)
        return {}


def infer_market_sentiment_from_rsi(
    price_df: pd.DataFrame,
    tickers: List[str],
    rsi_period: int = 14
) -> Tuple[float, str]:
    """
    Compute average RSI across all tickers and derive a coarse
    market sentiment flag.
    Returns (avg_rsi, sentiment_flag) where flag ∈ {"bear","bull","hot"}.
    """
    rsi_values: List[float] = []

    for t in tickers:
        try:
            df_t = price_df[t] if len(tickers) > 1 else price_df
            if df_t.empty or len(df_t) < rsi_period + 5:
                continue
            d = df_t.copy()
            d["RSI"] = compute_rsi(d["Close"], period=rsi_period)
            rsi_val = d["RSI"].iloc[-1]
            if pd.notna(rsi_val):
                rsi_values.append(float(rsi_val))
        except Exception as exc:
            logger.warning("RSI calc failed for %s: %s", t, exc)

    avg_rsi = sum(rsi_values) / len(rsi_values) if rsi_values else 50.0

    if avg_rsi < 45:
        sentiment_flag = "bear"
    elif avg_rsi > 70:
        sentiment_flag = "hot"
    else:
        sentiment_flag = "bull"

    return avg_rsi, sentiment_flag


# ---------------------------------------------------------------------------
# CORE STOCK ANALYSIS
# ---------------------------------------------------------------------------

def analyze_stock(
    symbol: str,
    df: pd.DataFrame,
    funds: float,
    risk_pct: float,
    market_sentiment_flag: str,
) -> Optional[Dict]:
    """
    Run technical + rudimentary fundamental analysis for a single symbol.

    Returns a dict suitable for display in the dashboard.
    """
    try:
        if df.empty or len(df) < 200:
            logger.info("Skipping %s due to insufficient history.", symbol)
            return None

        df = df.copy()

        # --- Technicals ---
        df["SMA200"] = df["Close"].rolling(200).mean()
        df["RSI"] = compute_rsi(df["Close"], period=14)
        df["ATR"] = compute_atr(df, period=14)

        curr = df.iloc[-1]
        price = float(curr["Close"])
        rsi = float(curr["RSI"])
        sma200 = float(curr["SMA200"])
        atr = float(curr["ATR"])
        recent_vol_mean = df["Volume"].tail(20).mean()
        rvol = float(curr["Volume"]) / recent_vol_mean if recent_vol_mean else 1.0

        # --- Fundamentals (best-effort via Yahoo) ---
        info = fetch_ticker_info(symbol)
        forward_pe = info.get("forwardPE", None)
        sector = info.get("sector") or info.get("industry")  # fallback
        long_name = info.get("longName") or info.get("shortName") or symbol

        revenue_growth = info.get("revenueGrowth", None)
        profit_margin = info.get("profitMargins", None)

        # --- Technical conviction scoring (simple heuristic) ---
        score = 0
        if price > sma200:
            score += 3
        if rvol > 1.8:
            score += 4
        if 45 < rsi < 68:
            score += 3

        status = "🟡 HOLD"
        if score >= 7 and rvol > 1.5:
            status = "🔥 BREAKOUT"
        elif rsi < 32:
            status = "💎 BUY DIP"

        # --- Valuation: forward P/E vs sector benchmark ---
        sector_bench = get_sector_benchmark(sector)
        valuation_view = "N/A"
        undervalued = False
        overvalued = False
        pe_vs_sector: Optional[float] = None

        if isinstance(forward_pe, (int, float)) and forward_pe > 0:
            pe_vs_sector = forward_pe / sector_bench if sector_bench else None
            if pe_vs_sector is not None:
                if pe_vs_sector < 0.8:
                    valuation_view = "⬇️ Undervalued vs sector"
                    undervalued = True
                elif pe_vs_sector > 1.2:
                    valuation_view = "⬆️ Overvalued vs sector"
                    overvalued = True
                else:
                    valuation_view = "⚖️ Fair vs sector"

        # --- Simple financial quality flag ---
        fin_quality = "Unknown"
        good_growth = isinstance(revenue_growth, (int, float)) and revenue_growth > 0.05
        ok_margin = isinstance(profit_margin, (int, float)) and profit_margin > 0.05
        weak_growth = isinstance(revenue_growth, (int, float)) and revenue_growth < 0
        weak_margin = isinstance(profit_margin, (int, float)) and profit_margin < 0

        if good_growth and ok_margin:
            fin_quality = "✅ Strong"
        elif weak_growth or weak_margin:
            fin_quality = "⚠️ Weak"
        elif (revenue_growth is not None) or (profit_margin is not None):
            fin_quality = "ℹ️ Mixed"

        # --- Position sizing ---
        # Basic ATR-based bet size: risk_pct of portfolio with 2*ATR stop.
        shares = 0
        if isinstance(atr, (int, float)) and atr > 0:
            dollar_risk_per_share = 2 * atr
            position_risk_dollars = funds * risk_pct
            shares = int(position_risk_dollars / dollar_risk_per_share) if dollar_risk_per_share > 0 else 0

        news_url = f"https://finance.yahoo.com/quote/{symbol}"

        # --- Entry / Exit levels ---
        entry_level = "—"
        exit_level = "—"

        if isinstance(atr, (int, float)) and atr > 0:
            base_entry = price
            stop_loss = price - 2 * atr
            take_profit = price + 3 * atr

            # ENTRY conditions
            entry_conditions: List[bool] = []

            # Valuation filter
            if undervalued:
                entry_conditions.append(True)
            elif not overvalued:
                entry_conditions.append(True)
            else:
                entry_conditions.append(False)

            # Trend/technical conditions
            entry_conditions.append(price >= sma200)      # uptrend
            entry_conditions.append(rsi < 70)             # avoid extreme overbought entries

            # Sentiment overlay
            if market_sentiment_flag == "hot":
                # In very hot tape, avoid new longs unless strong dip/undervalued.
                if not undervalued and rsi > 40:
                    entry_conditions.append(False)
            elif market_sentiment_flag == "bear":
                # In bear tape, require dip-buy signal.
                if status != "💎 BUY DIP":
                    entry_conditions.append(False)

            if all(entry_conditions):
                entry_level = f"Entry ~${base_entry:.2f} / SL ~${stop_loss:.2f}"
            else:
                entry_level = "No fresh entry (stand aside)"

            # EXIT logic
            exit_signals: List[str] = []
            if rsi > 70:
                exit_signals.append("RSI>70 (overbought)")
            if overvalued and price > 1.15 * sma200:
                exit_signals.append("Overvalued + extended vs 200d")
            if status == "🔥 BREAKOUT" and rvol > 2.5:
                exit_signals.append("Parabolic breakout – consider scaling out")

            if exit_signals:
                exit_level = f"TP ~${take_profit:.2f} | Tighten SL ({'; '.join(exit_signals)})"
            else:
                if price < sma200:
                    exit_level = "Below 200d – consider reducing / exiting on strength"
                else:
                    exit_level = "Hold bias – trail stop under recent swing lows"

        # --- Package result row ---
        result: Dict = {
            "Ticker": symbol,
            "Name": long_name,
            "Price": f"${price:.2f}",
            "Score": f"{score}/10",
            "RVOL": f"{rvol:.1f}x",
            "RSI": int(rsi) if pd.notna(rsi) else None,
            "Forward P/E": f"{forward_pe:.1f}" if isinstance(forward_pe, (int, float)) and forward_pe > 0 else "N/A",
            "Valuation": valuation_view,
            "Fin Quality": fin_quality,
            "Action": status,
            "Entry Level": entry_level,
            "Exit Level": exit_level,
            "Sizing": f"{shares} Shrs" if shares > 0 else "0 Shrs",
            "News": news_url,
        }

        return result

    except Exception as exc:
        logger.exception("Error analyzing %s: %s", symbol, exc)
        return None


# ---------------------------------------------------------------------------
# MAIN APP
# ---------------------------------------------------------------------------

def main() -> None:
    if not check_password():
        return

    st.title("🐋 Institutional Wealth Terminal 2026")

    # ------------- Sidebar Controls -------------
    with st.sidebar:
        funds: float = st.number_input("Portfolio $", value=100000.0, min_value=0.0, step=1000.0)
        risk_pct: float = st.slider("Risk % per Position", 0.5, 3.0, 1.5) / 100.0

        mode = st.radio("Scanner Mode", ["My Watchlist", "Momentum Hot Picks 🔥"])

        if mode == "My Watchlist":
            user_input = st.text_area("Symbols (comma-separated)", "NVDA,AAPL,TSLA,AMD")
            tickers = [t.strip().upper() for t in user_input.split(",") if t.strip()]
        else:
            tickers = get_hot_picks()

        run_scan = st.button("🚀 EXECUTE SCAN")

    # ------------- Data Fetch + Analytics -------------
    if run_scan or "results" not in st.session_state:
        if not tickers:
            st.warning("No symbols provided.")
            return

        with st.spinner("Processing Market Intelligence..."):
            price_df = fetch_price_history(tickers, period="1y")
            if price_df.empty:
                st.error("Failed to load price data. Check symbols or connectivity.")
                return

            avg_rsi, sentiment_flag = infer_market_sentiment_from_rsi(price_df, tickers)

            results: List[Dict] = []
            for symbol in tickers:
                try:
                    df_symbol = price_df[symbol] if len(tickers) > 1 else price_df
                except Exception:
                    logger.warning("Missing data for %s, skipping.", symbol)
                    continue

                row = analyze_stock(
                    symbol=symbol,
                    df=df_symbol,
                    funds=funds,
                    risk_pct=risk_pct,
                    market_sentiment_flag=sentiment_flag,
                )
                if row:
                    results.append(row)

            st.session_state.results = pd.DataFrame(results)

            # Correlation matrix for close prices over 6m
            try:
                corr_price_df = fetch_price_history(tickers, period="6mo")
                if corr_price_df.empty:
                    st.session_state.corr = pd.DataFrame()
                else:
                    close_df = corr_price_df["Close"] if "Close" in corr_price_df else corr_price_df
                    if isinstance(close_df.columns, pd.MultiIndex):
                        close_df.columns = close_df.columns.get_level_values(0)
                    corr = close_df.dropna(axis=1, how="all").corr()
                    st.session_state.corr = corr
            except Exception as exc:
                logger.warning("Correlation computation failed: %s", exc)
                st.session_state.corr = pd.DataFrame()

            st.session_state.avg_rsi = avg_rsi
            st.session_state.sentiment_flag = sentiment_flag

    # ------------- Top-level Metrics -------------
    tickers = tickers if "tickers" in locals() else []
    avg_rsi = st.session_state.get("avg_rsi", 50.0)
    sentiment_flag = st.session_state.get("sentiment_flag", "bull")

    if sentiment_flag == "bull":
        sentiment_label = "🐂 BULL"
    elif sentiment_flag == "hot":
        sentiment_label = "🛑 HOT"
    else:
        sentiment_label = "🐻 BEAR"

    c1, c2, c3 = st.columns(3)
    c1.metric("Assets Analyzed", len(tickers))
    c2.metric("Market Sentiment", sentiment_label, f"Avg RSI {avg_rsi:.1f}")
    c3.metric("Terminal Time", datetime.now().strftime("%H:%M"))

    # ------------- Dashboard -------------
    st.subheader("📋 Market Execution Dashboard")

    if "results" in st.session_state and not st.session_state.results.empty:
        st.dataframe(
            st.session_state.results,
            use_container_width=True,
            hide_index=True,
            column_config={
                "News": st.column_config.LinkColumn("Research"),
                "Forward P/E": st.column_config.TextColumn("Forward P/E"),
                "Entry Level": st.column_config.TextColumn("Entry / Risk"),
                "Exit Level": st.column_config.TextColumn("Exit / TP Guide"),
            },
        )
    else:
        st.info("No results to display. Run the scan to populate the dashboard.")

    # ------------- Risk / Correlation -------------
    st.subheader("🔥 Risk Correlation (Portfolio Diversity)")
    corr_df = st.session_state.get("corr", pd.DataFrame())
    if not corr_df.empty:
        st.dataframe(
            corr_df.style.background_gradient(cmap="RdYlGn", axis=None).format("{:.2f}"),
            use_container_width=True,
        )
    else:
        st.warning("Correlation data currently unavailable for this list.")

    # ------------- Disclaimers -------------
    st.caption(
        "Data Source: Yahoo Finance via yfinance (delayed, best-effort). "
        "Do not use this tool as a sole source for regulatory, accounting or "
        "trade-execution decisions. Always cross-check with your official market data feed."
    )


if __name__ == "__main__":
    main()
```



