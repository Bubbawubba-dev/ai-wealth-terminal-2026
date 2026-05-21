# TAB 2: TECHNICAL SENTIMENT
with tab_sentiment:
    st.subheader("Dynamic Fear & Greed Structural Proxies")
    selected_ticker = st.selectbox("Select Target Engine Asset:", universe)

    if not historical_data.empty:

        sentiment = calculate_advanced_sentiment(historical_data, selected_ticker)

        if sentiment["status"] == "Active":

            # --- METRICS ROW ---
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Aggregate Score", sentiment["score"], sentiment["label"])
            with col2:
                st.metric("RSI (14 Daily)", sentiment["metrics"]["rsi_14"])
            with col3:
                st.metric("Volatility Multiplier", f"{sentiment['metrics']['volatility_ratio']}x")

            # --- SENTIMENT GAUGE ---
            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=sentiment["score"],
                title={"text": "Sentiment Gauge (0–100)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#38bdf8"},
                    "steps": [
                        {"range": [0, 25], "color": "#1e3a8a"},
                        {"range": [25, 45], "color": "#0f766e"},
                        {"range": [45, 55], "color": "#475569"},
                        {"range": [55, 75], "color": "#ca8a04"},
                        {"range": [75, 100], "color": "#b91c1c"},
                    ],
                }
            ))
            st.plotly_chart(gauge_fig, use_container_width=True)

            # --- LOAD PRICE DATA ---
            ticker_df = historical_data[selected_ticker].dropna()
            close = ticker_df["Close"]
            high = ticker_df["High"]
            low = ticker_df["Low"]

            # --- CALCULATE RSI ---
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi_series = 100 - (100 / (1 + rs))

            # --- SMA20 ---
            sma20 = close.rolling(20).mean()

            # --- VOLATILITY RATIO ---
            tr = np.maximum((high - low),
                            np.maximum(abs(high - close.shift(1)),
                                       abs(low - close.shift(1))))
            atr5 = tr.rolling(5).mean()
            atr20 = tr.rolling(20).mean()
            vol_ratio_series = atr5 / atr20

            # --- PRICE CHART WITH SIGNAL ARROWS ---
            fig_price = go.Figure()
            fig_price.add_trace(go.Scatter(
                x=close.index, y=close,
                name="Close", line=dict(color="#38bdf8", width=2)
            ))
            fig_price.add_trace(go.Scatter(
                x=sma20.index, y=sma20,
                name="SMA20", line=dict(color="#f59e0b", dash="dash")
            ))

            # --- SIGNAL ARROWS ---
            buy_signals = []
            sell_signals = []

            for i in range(1, len(close)):
                # BUY
                if (
                    close.iloc[i] > sma20.iloc[i] and
                    close.iloc[i-1] <= sma20.iloc[i-1] and
                    rsi_series.iloc[i] > 50 and
                    vol_ratio_series.iloc[i] > 1.0
                ):
                    buy_signals.append((close.index[i], close.iloc[i]))

                # SELL
                if (
                    close.iloc[i] < sma20.iloc[i] and
                    close.iloc[i-1] >= sma20.iloc[i-1] or
                    rsi_series.iloc[i] < 45
                ):
                    sell_signals.append((close.index[i], close.iloc[i]))

            # Plot arrows
            for t, p in buy_signals:
                fig_price.add_annotation(
                    x=t, y=p, text="⬆ BUY",
                    showarrow=True, arrowhead=1,
                    font=dict(color="#22c55e")
                )

            for t, p in sell_signals:
                fig_price.add_annotation(
                    x=t, y=p, text="⬇ SELL",
                    showarrow=True, arrowhead=1,
                    font=dict(color="#ef4444")
                )

            fig_price.update_layout(
                title=f"{selected_ticker} — Price with Signals",
                template="plotly_dark", height=300
            )
            st.plotly_chart(fig_price, use_container_width=True)

            # --- BACKTEST ENGINE ---
            st.markdown("### 📈 Backtest Results (10–30 Day Swing Strategy)")

            returns = []
            position = None
            entry_price = None

            for i in range(1, len(close)):

                # ENTRY
                if (
                    position is None and
                    close.iloc[i] > sma20.iloc[i] and
                    rsi_series.iloc[i] > 50 and
                    vol_ratio_series.iloc[i] > 1.0
                ):
                    position = "LONG"
                    entry_price = close.iloc[i]

                # EXIT
                elif (
                    position == "LONG" and (
                        close.iloc[i] < sma20.iloc[i] or
                        rsi_series.iloc[i] < 45 or
                        vol_ratio_series.iloc[i] < 0.8
                    )
                ):
                    returns.append((close.iloc[i] - entry_price) / entry_price)
                    position = None
                    entry_price = None

            # Summary
            if returns:
                avg_return = np.mean(returns) * 100
                win_rate = (np.sum(np.array(returns) > 0) / len(returns)) * 100
                st.metric("Avg Trade Return", f"{avg_return:.2f}%")
                st.metric("Win Rate", f"{win_rate:.1f}%")
                st.metric("Total Trades", len(returns))
            else:
                st.info("Not enough signals to compute backtest.")

            # --- PROBABILITY MODEL ---
            st.markdown("### 🔮 Trend Continuation Probability")

            prob = (
                0.4 * (sentiment["metrics"]["rsi_14"] / 100) +
                0.4 * (max(0, sentiment["metrics"]["ma_deviation_pct"]) / 20) +
                0.2 * min(1.5, sentiment["metrics"]["volatility_ratio"]) / 1.5
            )

            probability = min(100, max(0, prob * 100))

            st.metric("Continuation Probability", f"{probability:.1f}%")

        else:
            st.error(f"Engine Fault: {sentiment['error']}")
