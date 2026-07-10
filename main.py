with tab_sentiment:
    st.subheader("Dynamic Fear & Greed Structural Proxies")

    if historical_data.empty:
        st.error("Historical data unavailable.")
    else:
        selected_ticker = st.selectbox("Select Target Engine Asset:", full_universe)

        if selected_ticker in historical_data.columns.get_level_values(0):

            engine_choice = st.toggle("Use Momentum Engine v2", value=True)

            # SENTIMENT ENGINE
            sentiment = calculate_advanced_sentiment(historical_data, selected_ticker)

            if sentiment["status"] == "Active":
                ticker_df = _normalize_ohlcv(historical_data[selected_ticker]).dropna()
                close = ticker_df["close"]
                high = ticker_df["high"]
                low = ticker_df["low"]

                # MOMENTUM ENGINE (v2 or legacy)
                if engine_choice:
                    daily = fetch_ohlcv_daily(selected_ticker)
                    results = analyze_ticker(
                        daily=daily,
                        h4=None,
                        h1=None,
                        equity=100_000,
                        cfg=EngineConfig(),
                    )

                    mqs = results["mqs"].iloc[-1]
                    phase = results["phase"].iloc[-1]
                    entry = results["entry_signal"].iloc[-1]
                    exit_ = results["exit_signal"].iloc[-1]
                    long_ok = results["long_ok"].iloc[-1]
                    narrative = results["narrative"].iloc[-1]
                else:
                    # Legacy placeholder logic
                    mqs = sentiment["score"]
                    phase = "Legacy Engine"
                    entry = False
                    exit_ = False
                    long_ok = False
                    narrative = "Legacy engine active — no narrative available."
