# TAB 1 — Explosive Short-Term Breakout Intelligence Engine
with tab_momentum:

    st.subheader("Explosive Short-Term Breakout Scanner")

    # -----------------------------
    # Helper Functions
    # -----------------------------

    def calculate_breakout_score(row):
        score = 0

        # Price vs SMA20/SMA50
        if row["close"] > row["sma20"]:
            score += 15
        if row["close"] > row["sma50"]:
            score += 15

        # Volume expansion
        if row["volume"] > row["vol_avg20"] * 1.5:
            score += 20
        elif row["volume"] > row["vol_avg20"]:
            score += 10

        # RSI acceleration
        if row["rsi_change"] > 2:
            score += 15
        elif row["rsi_change"] > 0:
            score += 5

        # Breakout above recent high
        if row["close"] > row["high_20d"]:
            score += 20

        return min(score, 100)


    def classify_trend_maturity(row):
        if row["sma20"] > row["sma50"] and row["close"] < row["sma20"] * 1.03:
            return "Early"
        if row["sma20"] > row["sma50"] and row["close"] < row["sma20"] * 1.08:
            return "Mid"
        if row["close"] > row["sma20"] * 1.10:
            return "Late"
        return "Exhaustion"


    def classify_momentum_cluster(row):
        if row["breakout_score"] > 70 and row["volume"] > row["vol_avg20"] * 1.5:
            return "Explosive"
        if row["breakout_score"] > 50:
            return "Emerging"
        if row["rsi_change"] < 0:
            return "Cooling"
        if row["close"] > row["sma20"] * 1.12:
            return "Overextended"
        return "Reversal Watch"


    def calculate_scenario_probabilities(row):
        continuation = 40
        pullback = 30
        reversal = 30

        if row["breakout_score"] > 70:
            continuation += 20
            pullback -= 10

        if row["trend_maturity"] == "Late":
            pullback += 20
            continuation -= 10

        if row["momentum_cluster"] == "Cooling":
            reversal
