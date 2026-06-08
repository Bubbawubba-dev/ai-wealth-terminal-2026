@st.cache_data(ttl=300)
def fetch_intraday_snapshot(tickers, interval="5m", days=3):
    if not tickers:
        return {}
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    try:
        data = yf.download(
            tickers,
            start=start_date,
            interval=interval,
            group_by="ticker",
            progress=False,
        )
        if isinstance(data.columns, pd.MultiIndex):
            out = {}
            for t in data.columns.get_level_values(0).unique():
                out[t] = data[t].dropna()
            return out
        else:
            # single ticker case
            return {tickers[0]: data.dropna()}
    except Exception:
        return {}

def compute_market_shock_index(index_df, vix_df=None, breadth_pct=None):
    if index_df is None or index_df.empty:
        return 50  # neutral fallback

    open_today = index_df["Open"].iloc[0]
    last_close = index_df["Close"].iloc[-1]
    intraday_ret = (last_close - open_today) / open_today * 100

    shock_price = np.interp(intraday_ret, [-4, -2, 0], [100, 80, 40])

    shock_vol = 50
    if vix_df is not None and not vix_df.empty:
        vix_change = (vix_df["Close"].iloc[-1] - vix_df["Close"].iloc[-2]) / vix_df["Close"].iloc[-2] * 100
        shock_vol = np.interp(vix_change, [0, 10, 30], [40, 70, 95])

    shock_breadth = 50
    if breadth_pct is not None:
        shock_breadth = np.interp(breadth_pct, [20, 40, 60], [95, 70, 40])

    composite = 0.4 * shock_price + 0.35 * shock_vol + 0.25 * shock_breadth
    return int(np.clip(composite, 0, 100))

def compute_ticker_shock(intraday_df, daily_tail_df):
    if intraday_df is None or intraday_df.empty or daily_tail_df is None or daily_tail_df.empty:
        return {
            "intraday_return_pct": 0.0,
            "daily_vol_pct": 0.0,
            "shock_z": 0.0,
            "shock_score": 50,
        }

    open_today = intraday_df["Open"].iloc[0]
    last_close = intraday_df["Close"].iloc[-1]
    intraday_ret = (last_close - open_today) / open_today * 100

    daily_close = daily_tail_df["Close"]
    daily_ret = daily_close.pct_change().dropna()
    vol = daily_ret.std() * 100 if len(daily_ret) > 5 else 1.0

    shock_z = intraday_ret / (vol if vol > 0 else 1.0)
    shock_score = np.interp(shock_z, [-3, -2, -1, 0], [100, 80, 65, 45])

    return {
        "intraday_return_pct": round(intraday_ret, 2),
        "daily_vol_pct": round(vol, 2),
        "shock_z": round(shock_z, 2),
        "shock_score": int(np.clip(shock_score, 0, 100)),
    }
