import streamlit as st
import yfinance as yf
import pandas as pd
import requests
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timezone

# --- 1. CONFIG ---
st.set_page_config(page_title="Wealth Terminal v11.3", layout="wide")

# --- 2. SCRAPER ---
@st.cache_data(ttl=3600)
def get_micro_cap_universe():
    try:
        url = "wikipedia.org"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        df_list = pd.read_html(response.text)
        for df in df_list:
            df.columns = [str(c).strip() for c in df.columns]
            col_candidates = [col for col in df.columns if any(x in col.upper() for x in ['TICKER', 'SYMBOL'])]
            if col_candidates:
                target_col = col_candidates[0]
                tickers = df[target_col].dropna().astype(str).tolist()
                clean_tickers = []
                for t in tickers:
                    token = t.split(':')[-1].replace(')', '').strip().upper()
                    if token.isalpha() and len(token) <= 5:
                        clean_tickers.append(token)
                if clean_tickers:
                    return list(set(clean_tickers))[:15]
    except:
        pass
    return ["MRAM", "ASTS", "HIMS", "QUBT", "BZFD", "HUT", "FLEX", "VCYT", "VECO", "IONQ"]

# --- 2B. TOP 10 MOMENTUM STOCKS FETCHER ---
@st.cache_data(ttl=86400)  # Cache for 24 hours (daily refresh)
def get_top_10_momentum_stocks():
    """
    Automatically fetches top 10 most volatile/momentum stocks from a predefined universe.
    Uses volume velocity and recent price momentum as selection criteria.
    """
    candidate_universe = [
        "MRAM", "ASTS", "HIMS", "QUBT", "BZFD", "HUT", "FLEX", "VCYT", "VECO", "IONQ",
        "RKLB", "KTOS", "CYBR", "GNK", "PHYS", "CEG", "NVDA", "MSFT", "AAPL", "TSLA",
        "AMD", "SOFI", "PLTR", "UPST", "U", "COIN", "RIOT", "MSTR", "SQ", "DKNG"
    ]
    
    momentum_scores = []
    
    for ticker in candidate_universe:
        try:
            ticker_obj = yf.Ticker(ticker)
            data = ticker_obj.history(period="3mo")
            
            if data is not None and not data.empty and len(data) >= 20:
                # Calculate momentum metrics
                close_price = float(data['Close'].iloc[-1])
                
                # 5-day momentum
                momentum_5d = (close_price / data['Close'].iloc[-5] - 1) * 100 if len(data) >= 5 else 0
                
                # Volume velocity (last 3 days avg vs 20-day avg)
                vol_recent = data['Volume'].tail(3).mean()
                vol_baseline = data['Volume'].tail(20).mean()
                xvol = vol_recent / vol_baseline if vol_baseline > 0 else 1.0
                
                # Average True Range for volatility
                atr = (data['High'] - data['Low']).rolling(14).mean().iloc[-1] if len(data) >= 14 else 0
                
                # Composite momentum score
                momentum_score = (momentum_5d * 0.4) + (xvol * 25 * 0.4) + (atr / close_price * 100 * 0.2)
                
                momentum_scores.append({
                    "Ticker": ticker,
                    "Price": round(close_price, 2),
                    "Momentum_5D": round(momentum_5d, 2),
                    "xVOL": round(xvol, 2),
                    "Score": round(momentum_score, 2)
                })
        except:
            pass
    
    # Sort by composite score and return top 10
    if momentum_scores:
        df_momentum = pd.DataFrame(momentum_scores)
        return df_momentum.nlargest(10, 'Score')['Ticker'].tolist()
    
    return ["MRAM", "ASTS", "HIMS", "QUBT", "BZFD", "HUT", "FLEX", "VCYT", "VECO", "IONQ"]

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

# --- 4. ANALYTICS ENGINE WITH THREE EXECUTION RULES ---
def analyze_stock(symbol, df, ticker_obj, funds, risk, enable_analyst_picks, history_floor=60, aggressive_mode=False):
    try:
        if df is None or len(df) < history_floor:
            return None
       
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            return None

        # Fetch Financial Parameters Safely
        info = ticker_obj.info if hasattr(ticker_obj, 'info') else {}
        operating_margin = info.get('operatingMargins', 0.0)
        return_on_assets = info.get('returnOnAssets', 0.0)
        operating_margin = 0.0 if operating_margin is None else float(operating_margin)
        return_on_assets = 0.0 if return_on_assets is None else float(return_on_assets)

        # Technical Engine Metrics
        has_macro_history = len(df) >= 200
        df['SMA200'] = df['Close'].rolling(200).mean() if has_macro_history else df['Close'].mean()
        df['SMA50'] = df['Close'].rolling(50).mean() if len(df) >= 50 else df['Close'].mean()
       
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
        df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
       
        curr = df.iloc[-1]
        price, rsi, sma50, sma200, atr = float(curr['Close']), float(curr['RSI']), float(curr['SMA50']), float(curr['SMA200']), float(curr['ATR'])
       
        # Volume Velocity Analytics
        short_term_vol = df['Volume'].tail(3).mean()
        historical_base = df['Volume'].tail(60).mean()
        xvol = float(short_term_vol / historical_base if historical_base > 0 else 1.0)
       
        dist_from_sma50 = float((price / sma50) - 1 if sma50 > 0 else 0.0)
        suggested_entry = float(df['High'].tail(5).max())
       
        # Research Wizard Parameters
        chg_4w = float((price / df['Close'].iloc[-21]) - 1.0) if len(df) >= 21 else 0.0
        high_52w = float(df['High'].tail(252).max() if len(df) >= 252 else df['High'].max())
        ratio_52w = float(price / high_52w if high_52w > 0 else 0.0)

        eps_revision_momentum = 0.0
        try:
            earn_hist = ticker_obj.get_earnings_history() if hasattr(ticker_obj, 'get_earnings_history') else None
            if earn_hist is not None and not earn_hist.empty:
                df_earn = pd.DataFrame(earn_hist).dropna(subset=['epsActual', 'epsEstimate'])
                if not df_earn.empty:
                    df_earn = df_earn.tail(4)
                    avg_actual = df_earn['epsActual'].mean()
                    avg_estimate = df_earn['epsEstimate'].mean()
                    if abs(avg_estimate) > 0.01:
                        eps_revision_momentum = float((avg_actual - avg_estimate) / abs(avg_estimate))
        except:
            pass

        zacks_score = 3
        if rsi >= 65 and xvol >= 2.0: zacks_score -= 1
        if eps_revision_momentum > 0.05: zacks_score -= 1  
        if eps_revision_momentum < -0.05: zacks_score += 1
        if rsi > 82 or dist_from_sma50 > 0.35: zacks_score += 1
        zacks_rank = int(max(1, min(5, zacks_score)))

        # Technical Scoring Matrix - AGGRESSIVE MODE ADJUSTMENTS
        score = 0
        is_above_sma200 = price > sma200 if has_macro_history else True
        if is_above_sma200: score += 3  
        if 65 <= rsi <= 82: score += 5  
        elif rsi > 82: score -= 2 if not aggressive_mode else 0  # More lenient in aggressive mode
        if xvol >= 4.0: score += 4  
        elif xvol >= 2.0: score += 2  
        if dist_from_sma50 > 0.55: score -= 2 if not aggressive_mode else -1  # More lenient extension in aggressive mode
           
        base_status = "🟡 MONITOR"
        reason = "Awaiting Momentum Confirmation"
        
        # AGGRESSIVE MODE: Lower thresholds for swings
        if aggressive_mode:
            if score >= 4 and xvol >= 1.5:
                base_status = "🔥 BUY"
                reason = "High Volatility Swing Opportunity"
        else:
            if score >= 6 and xvol >= 2.0 and dist_from_sma50 < 0.50:
                base_status = "🔥 BUY"
                reason = "Explosive Volume Breakout Run"

        vol_is_drying_up = float(curr['Volume']) < df['Volume'].tail(5).mean()
        is_overextended = dist_from_sma50 > 0.50
        has_broken_out_5d = price >= suggested_entry

        status = base_status
        if base_status == "🔥 BUY":
            if aggressive_mode:
                # In aggressive mode, be more permissive with extended moves
                if vol_is_drying_up and not has_broken_out_5d:
                    status = "⏳ BASE FORMING (LOW VOL)"
                    reason = "Volume consolidating at support levels."
                elif not has_broken_out_5d:
                    status = "🟡 SETTING UP"
                    reason = "Waiting for intra-day momentum confirmation."
                else:
                    status = "🚀 EXECUTE SWING SCALP"
                    reason = "Aggressive momentum trade ready."
            else:
                if is_overextended:
                    status = "⏳ COOLING OFF (OVEREXTENDED)"
                    reason = "Rule 2: Extended >50% from SMA50."
                elif vol_is_drying_up and not has_broken_out_5d:
                    status = "⏳ BASE FORMING (LOW VOL)"
                    reason = "Rule 1 & 3: Volume drying up at highs."
                elif not has_broken_out_5d:
                    status = "🟡 SETTING UP"
                    reason = "Rule 3: Waiting for cross above 5-Day High."
                else:
                    status = "🚀 EXECUTE ACTIVE BUY"
                    reason = "All 3 Execution Rules Cleared."

        # Risk Management Controls
        initial_stop_price = float(price - (atr * 1.8))
        trailing_stop_floor = float(price - (atr * 1.5))  
        take_profit_target = float(price + (atr * 2.5))  
       
        horizon = "⏳ WATCHLIST"
        if "BUY" in status or "EXECUTE" in status or "SWING" in status:
            horizon = "⚡ SHORT-TERM SWING" if xvol >= 3.0 else "💎 LONG-TERM HOLD"
        elif price <= initial_stop_price:
            status = "🛑 HARD STOP EXCEEDED"
            horizon = "❌ EXIT POSITION"
            reason = "Volatility Floor Trailed Out"

        daytrade_target = float(price + (atr * 1.5))
        daytrade_stop = float(price - (atr * 1.0))

        # Trend Horizon Calculator Engine
        base_days = 0
        holding_guide = "⚡ DAY TRADE ONLY"
        try:
            local_high_boundary = high_52w * 0.88
            history_slice_len = min(len(df), 90)
            historical_closes = df['Close'].tail(history_slice_len).tolist()
            for p_close in reversed(historical_closes):
                if p_close >= local_high_boundary:
                    base_days += 1
                else:
                    break
            if has_macro_history and sma50 > sma200 and price > sma200:
                if base_days >= 45:
                    holding_guide = "🐋 CORE MACRO HOLD (Months)"
                elif base_days >= 15:
                    holding_guide = "💎 MULTI-WEEK SWING (Weeks)"
                else:
                    holding_guide = "⚡ SHORT SWING TACTICAL"
            else:
                holding_guide = "⚡ DAY TRADE ONLY (No Macro Floor)"
        except:
            pass

        return {
            "Ticker": symbol, "Price": round(price, 2), "Score": f"{score}/10",
            "Action": status, "Horizon Allocation": horizon, "Trigger Reason": reason,
            "Ext%": f"{dist_from_sma50*100:.1f}%", "RSI": int(rsi), "xVOL Velocity": f"{xvol:.1f}x",
            "Initial Stop Floor": round(initial_stop_price, 2), "Dynamic Trailing Stop": round(trailing_stop_floor, 2),
            "Take Profit Target": round(take_profit_target, 2),    
            "Sizing": f"{int((funds * risk)/(price - initial_stop_price)) if (price - initial_stop_price)>0 else 0} Shrs",
            "Chg_4W_Raw": float(chg_4w), "Ratio_52W_Raw": float(ratio_52w), "Zacks_Rank": int(zacks_rank),
            "EPS_Revision_Delta": float(eps_revision_momentum), "Operating_Margin": operating_margin, "ROA": return_on_assets,
            "DT_Trigger": "ACTIVE" if has_broken_out_5d else "STAGED", "DT_Target": round(daytrade_target, 2), "DT_Stop": round(daytrade_stop, 2),
            "Base_Duration_Days": int(base_days), "Holding_Horizon_Guide": holding_guide, "Score_Internal_Num": int(score)
        }
    except:
        return None

# --- 5. DATA & UI ENVIRONMENT ---
if check_password():
    st.title("🐋 Institutional Micro-Cap Terminal v11.3")
   
    with st.sidebar:
        st.header("⚙️ Capital Allocator")
        funds = st.number_input("Portfolio Target Deployment $", value=100000)
        risk = st.slider("Risk Per Trade Tolerance %", 0.5, 3.0, 1.5) / 100
       
        st.write("---")
        st.header("🔍 Index Feed Filters")
        enable_analyst_picks = st.checkbox("Enable Velocity Overlays", value=True)
        feed_mode = st.radio("Active Engine Feed Source", ["Scrape Automated Micro-Cap Index 🚀", "Manual Watchlist Tickers 📋"])
       
        if "Manual Watchlist Tickers 📋" in feed_mode:
            user_input = st.text_area("Watchlist Input", "MRAM,ASTS,HIMS,QUBT,NVDA,MSFT,CEG,PHYS,CYBR,GNK,AAPL,OXY,BAC,RKLB,ONDS,KTOS")
            t_list = [t.strip().upper() for t in user_input.replace("\n", ",").split(",") if t.strip()]
        else:
            with st.spinner("Scraping index..."):
                t_list = get_micro_cap_universe()
            st.info(f"Scraped Tickers Locked: {', '.join(t_list)}")
           
        st.write("---")
        st.header("📊 Breakout Ranking Priority")
        sort_by = st.selectbox("Rank Breakout Priority By:", ["Volume Velocity (xVOL)", "Extension Level (Ext%)", "Technical Score"])
        sort_order = st.radio("Sort Order Direction:", ["Highest First 📈", "Lowest First 📉"])
        ascending_bool = sort_order == "Lowest First 📉"
       
        run = st.button("🚀 EXECUTE ALPHA VELOCITY SWEEP")

    # Initialize session state for bulk_data if not present
    if "bulk_data" not in st.session_state:
        st.session_state.bulk_data = {}

    # --- TAB 6 AUTO-LOAD SECTION ---
    auto_load_tab6 = False
    if "new_swings_results" not in st.session_state or st.session_state.new_swings_results.empty:
        auto_load_tab6 = True

    if auto_load_tab6:
        with st.spinner("⚡ Auto-loading Top 10 Daily Momentum Stocks for Tab 6..."):
            try:
                top_10_tickers = get_top_10_momentum_stocks()
                res_list_auto_tab6 = []
                clean_ticker_data_tab6 = {}
                
                for t in top_10_tickers:
                    try:
                        ticker_obj = yf.Ticker(t)
                        ticker_data = ticker_obj.history(period="2y")
                        if ticker_data is not None and not ticker_data.empty:
                            if isinstance(ticker_data.columns, pd.MultiIndex):
                                ticker_data.columns = ticker_data.columns.get_level_values(0)
                            clean_ticker_data_tab6[t] = ticker_data
                            
                            res_swings = analyze_stock(t, ticker_data, ticker_obj, funds, risk, enable_analyst_picks, history_floor=20, aggressive_mode=True)
                            if res_swings: 
                                res_list_auto_tab6.append(res_swings)
                    except:
                        pass
                
                if res_list_auto_tab6:
                    raw_swings_df = pd.DataFrame(res_list_auto_tab6)
                    raw_swings_df['RVOL_num'] = raw_swings_df['xVOL Velocity'].astype(str).str.replace('x', '', regex=False).astype(float) if 'xVOL Velocity' in raw_swings_df.columns else 1.0
                    sorted_swings = raw_swings_df.sort_values(by="RVOL_num", ascending=False)
                    st.session_state.new_swings_results = sorted_swings.drop(columns=['RVOL_num', 'Score_Internal_Num'], errors='ignore')
                    
                    # FIXED: Properly merge bulk_data dictionaries
                    if "bulk_data" in st.session_state:
                        st.session_state.bulk_data.update(clean_ticker_data_tab6)
                    else:
                        st.session_state.bulk_data = clean_ticker_data_tab6
            except Exception as e:
                st.warning(f"⚠️ Could not auto-load Tab 6 data: {str(e)}")

    if run or "results" not in st.session_state:
        res_list_strict = []
        res_list_new_swings = []
        clean_ticker_data = {}
       
        for t in t_list:
            try:
                ticker_obj = yf.Ticker(t)
                ticker_data = ticker_obj.history(period="2y")
                if ticker_data is not None and not ticker_data.empty:
                    if isinstance(ticker_data.columns, pd.MultiIndex):
                        ticker_data.columns = ticker_data.columns.get_level_values(0)
                    clean_ticker_data[t] = ticker_data
                   
                    res_strict = analyze_stock(t, ticker_data, ticker_obj, funds, risk, enable_analyst_picks, history_floor=60)
                    if res_strict: res_list_strict.append(res_strict)
                   
                    res_swings = analyze_stock(t, ticker_data, ticker_obj, funds, risk, enable_analyst_picks, history_floor=20)
                    if res_swings: res_list_new_swings.append(res_swings)
            except:
                pass
               
        if res_list_strict:
            raw_df = pd.DataFrame(res_list_strict)
            raw_df['RVOL_num'] = raw_df['xVOL Velocity'].astype(str).str.replace('x', '', regex=False).astype(float) if 'xVOL Velocity' in raw_df.columns else 1.0
            raw_df['Ext_num'] = raw_df['Ext%'].astype(str).str.replace('%', '', regex=False).astype(float) if 'Ext%' in raw_df.columns else 0.0
            raw_df['Score_num'] = raw_df['Score_Internal_Num'].astype(int) if 'Score_Internal_Num' in raw_df.columns else 0
           
            sort_map = {"Volume Velocity (xVOL)": "RVOL_num", "Extension Level (Ext%)": "Ext_num", "Technical Score": "Score_num"}
            target_column = sort_map.get(sort_by, "RVOL_num")
            sorted_df = raw_df.sort_values(by=target_column, ascending=ascending_bool)
            st.session_state.results = sorted_df.drop(columns=['RVOL_num', 'Ext_num', 'Score_num', 'Score_Internal_Num'], errors='ignore')
        else:
            st.session_state.results = pd.DataFrame(columns=["Ticker", "Price", "Score", "Action", "Horizon Allocation", "Trigger Reason", "Ext%", "RSI", "xVOL Velocity", "Initial Stop Floor", "Dynamic Trailing Stop", "Take Profit Target", "Sizing"])

        if res_list_new_swings:
            raw_swings_df = pd.DataFrame(res_list_new_swings)
            raw_swings_df['RVOL_num'] = raw_swings_df['xVOL Velocity'].astype(str).str.replace('x', '', regex=False).astype(float) if 'xVOL Velocity' in raw_swings_df.columns else 1.0
            sorted_swings = raw_swings_df.sort_values(by="RVOL_num", ascending=False)
            st.session_state.new_swings_results = sorted_swings.drop(columns=['RVOL_num', 'Score_Internal_Num'], errors='ignore')
        else:
            st.session_state.new_swings_results = pd.DataFrame(columns=["Ticker", "Price", "Score", "Action"])
           
        st.session_state.bulk_data = clean_ticker_data

    # --- 6-TAB MATRIX NAVIGATION ENVIRONMENT ---
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📋 Execution Dashboard",
        "📈 Technical Visualizer Canvas",
        "🔬 Research Wizard Matrix",
        "🌌 Blue Sky Finder",
        "👥 Investor Alpha Network",
        "🔥 New Swings (Aggressive)"
    ])
   
    with tab1:
        st.subheader(f"Micro-Cap Breakout Execution Matrix (Sorted by {sort_by})")
        if not st.session_state.results.empty:
            exclude_internal = ["Chg_4W_Raw", "Ratio_52W_Raw", "Zacks_Rank", "EPS_Revision_Delta", "Operating_Margin", "ROA", "DT_Trigger", "DT_Target", "DT_Stop", "Base_Duration_Days", "Holding_Horizon_Guide"]
            display_cols = [c for c in st.session_state.results.columns if c not in exclude_internal]
            st.dataframe(st.session_state.results[display_cols], use_container_width=True, hide_index=True)
        else:
            st.info("Execute scanner sweeps to track pipeline data.")

    with tab2:
        valid_selections = [t for t in t_list if t in st.session_state.bulk_data] if "bulk_data" in st.session_state else []
        if valid_selections:
            sel = st.radio("Asset Pivot View:", valid_selections, horizontal=True)
            if sel and sel in st.session_state.bulk_data:
                df_plot = st.session_state.bulk_data[sel].copy()
                df_plot.index = pd.to_datetime(df_plot.index)
               
                 # FIXED SUBPLOT DICTIONARY SYNTAX
                fig = make_subplots(specs=[[{"secondary_y": True}]])
               
                fig.add_trace(go.Candlestick(x=df_plot.index, open=df_plot['Open'], high=df_plot['High'], low=df_plot['Low'], close=df_plot['Close'], name="Price"), secondary_y=False)
                if 'SMA200' in df_plot.columns:
                    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['SMA200'], line=dict(color='gold', width=2), name='SMA 200'), secondary_y=False)
                if 'SMA50' in df_plot.columns:
                    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['SMA50'], line=dict(color='cyan', width=1), name='SMA 50'), secondary_y=False)
               
                 # REVERTED VOLUME LOOK: Streamlined trace line presentation overlay [INDEX]
                fig.add_trace(go.Scatter(
                    x=df_plot.index,
                    y=df_plot['Volume'],
                    mode='lines',
                    line=dict(color='rgba(255, 165, 0, 0.45)', width=1.8),
                    name='Volume Line'
                ), secondary_y=True)
               
                fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=600, margin=dict(t=20, b=20, l=20, r=20))
                fig.update_yaxes(title_text="<b>Stock Share Price ($)</b>", color="white", secondary_y=False)
                fig.update_yaxes(title_text="<b>Institutional Liquidity Volume Curve</b>", color="orange", secondary_y=True)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Execute scan sweeps to display charting visuals.")

    with tab3:
        st.header("🔬 Institutional Factor Screen Layer")
        if not st.session_state.results.empty and "Chg_4W_Raw" in st.session_state.results.columns:
            df_wizard = st.session_state.results.copy()
            f1 = (df_wizard['Chg_4W_Raw'] >= 0.10) & (df_wizard['Chg_4W_Raw'] <= 0.20)
            f2 = df_wizard['Ratio_52W_Raw'] >= 0.90
            passed_stocks = df_wizard[f1 & f2].copy()
            failed_stocks = df_wizard[~(f1 & f2)].copy()
           
            if len(passed_stocks) > 0:
                passed_stocks['4W % Change'] = (passed_stocks['Chg_4W_Raw'] * 100).round(2)
                passed_stocks['Proximity to 52W High'] = passed_stocks['Ratio_52W_Raw'].round(3)
                passed_stocks['Revision Delta %'] = (passed_stocks['EPS_Revision_Delta'] * 100).round(1)
            if len(failed_stocks) > 0:
                failed_stocks['4W % Change'] = (failed_stocks['Chg_4W_Raw'] * 100).round(2)
                failed_stocks['Proximity to 52W High'] = failed_stocks['Ratio_52W_Raw'].round(3)
                failed_stocks['Revision Delta %'] = (failed_stocks['EPS_Revision_Delta'] * 100).round(1)
           
            col_l, col_r = st.columns([0.4, 0.6])
            with col_l:
                st.subheader("Passed Strategic Screen")
                if len(passed_stocks) > 0:
                    st.dataframe(passed_stocks[["Ticker", "Price", "4W % Change", "Proximity to 52W High", "Revision Delta %", "Zacks_Rank"]], use_container_width=True, hide_index=True)
                else:
                    st.warning("No assets match all parameters simultaneously.")
            with col_r:
                st.subheader("Analyst Trend Matrix Visualization")
                fig_wiz = make_subplots(specs=[[{"secondary_y": True}]])
                if len(failed_stocks) > 0:
                    fig_wiz.add_trace(go.Bar(x=failed_stocks['Ticker'], y=failed_stocks['Revision Delta %'].astype(float), name='Excluded: Rev Delta', marker_color='rgba(255, 99, 132, 0.2)'), secondary_y=False)
                    fig_wiz.add_trace(go.Scatter(x=failed_stocks['Ticker'], y=failed_stocks['Proximity to 52W High'].astype(float), mode='markers', name='Excluded: 52W Ratio', marker=dict(color='rgba(255, 99, 132, 0.4)', size=8)), secondary_y=True)
                if len(passed_stocks) > 0:
                    fig_wiz.add_trace(go.Bar(x=passed_stocks['Ticker'], y=passed_stocks['Revision Delta %'].astype(float), name='PASSED: Rev Delta', marker_color='#00FFCC'), secondary_y=False)
                    fig_wiz.add_trace(go.Scatter(x=passed_stocks['Ticker'], y=passed_stocks['Proximity to 52W High'].astype(float), mode='markers', name='PASSED: 52W Ratio', marker=dict(color='#FF6B35', size=10)), secondary_y=True)
                fig_wiz.update_layout(template="plotly_dark", height=550, title_text="Analyst Consensus Revision Overlays", xaxis_title="Ticker")
                st.plotly_chart(fig_wiz, use_container_width=True)

    with tab4:
        st.header("🌌 Blue Sky Breakout Engine")
        if not st.session_state.results.empty and "Ratio_52W_Raw" in st.session_state.results.columns:
            df_sky = st.session_state.results.copy()
            df_sky['RVOL_num'] = df_sky['xVOL Velocity'].astype(str).str.replace('x', '', regex=False).astype(float)
           
            gate_proximity = df_sky['Ratio_52W_Raw'] >= 0.96
            gate_fundamental = df_sky['RVOL_num'] >= 1.5
            passed_sky = df_sky[gate_proximity & gate_fundamental].copy()
           
            if not passed_sky.empty:
                st.success(f"🔥 {len(passed_sky)} Micro-Caps Found Coiled Within 4% of All-Time Highs")
                passed_sky['52W High Proximity'] = passed_sky['Ratio_52W_Raw'].round(3)
                st.dataframe(passed_sky[["Ticker", "Price", "52W High Proximity", "Base_Duration_Days", "Holding_Horizon_Guide", "DT_Trigger", "DT_Target", "DT_Stop", "Sizing"]].rename(columns={"Base_Duration_Days": "Days", "Holding_Horizon_Guide": "Horizon", "DT_Trigger": "Trigger", "DT_Target": "PT", "DT_Stop": "SL"}), use_container_width=True, hide_index=True)
            else:
                st.warning("Zero micro-cap assets currently match the combined 0.96 high proximity gate.")
        else:
            st.info("Execute scanner sweeps to populate the blue sky momentum breakout matrices.")

    with tab5:
        st.header("👥 Institutional Whale Conviction Matrix Map")
        network_data = [
            {"Ticker": "AAPL", "Investor Entity": "Berkshire Hathaway", "Macro Thesis Sector": "Consumer Ecosystem", "Allocation Tier": "Core Asset"},
            {"Ticker": "OXY", "Investor Entity": "Berkshire Hathaway", "Macro Thesis Sector": "Permian Basin Energy", "Allocation Tier": "Acquisition Block"},
            {"Ticker": "BAC", "Investor Entity": "Berkshire Hathaway", "Macro Thesis Sector": "Banking Infrastructure", "Allocation Tier": "Yield Engine"},
            {"Ticker": "PHYS", "Investor Entity": "Michael Burry (Scion)", "Macro Thesis Sector": "Gold Bullion", "Allocation Tier": "Core Asset"},
            {"Ticker": "CYBR", "Investor Entity": "Michael Burry (Scion)", "Macro Thesis Sector": "Cybersecurity", "Allocation Tier": "Tactical Growth"},
            {"Ticker": "GNK", "Investor Entity": "Michael Burry (Scion)", "Macro Thesis Sector": "Marine Shipping", "Allocation Tier": "Asymmetric Cyclical"},
            {"Ticker": "MSFT", "Investor Entity": "L. Aschenbrenner Thesis", "Macro Thesis Sector": "Frontier Labs", "Allocation Tier": "Core Asset"},
            {"Ticker": "NVDA", "Investor Entity": "L. Aschenbrenner Thesis", "Macro Thesis Sector": "GPU Acceleration", "Allocation Tier": "Core Asset"},
            {"Ticker": "CEG", "Investor Entity": "L. Aschenbrenner Thesis", "Macro Thesis Sector": "Nuclear Grid Scaling", "Allocation Tier": "Alpha Layer"}
        ]
        df_network = pd.DataFrame(network_data)
       
        # UNIFIED GLOBAL DATA CACHING POOL
        combined_scanned_pool = pd.DataFrame()
        if not st.session_state.results.empty:
            combined_scanned_pool = pd.concat([combined_scanned_pool, st.session_state.results], ignore_index=True)
        if not st.session_state.new_swings_results.empty:
            combined_scanned_pool = pd.concat([combined_scanned_pool, st.session_state.new_swings_results], ignore_index=True)
           
        if not combined_scanned_pool.empty:
            combined_scanned_pool = combined_scanned_pool.drop_duplicates(subset=["Ticker"])
            target_tickers = df_network["Ticker"].tolist()
            live_matrix_match = combined_scanned_pool[combined_scanned_pool["Ticker"].isin(target_tickers)].copy()
           
            if not live_matrix_match.empty:
                available_cols = [c for c in ["Ticker", "Price", "Score", "Action", "Horizon Allocation", "xVOL Velocity", "Initial Stop Floor", "Take Profit Target"] if c in live_matrix_match.columns]
                live_matrix_match = live_matrix_match[available_cols]
                final_mapped_network_df = pd.merge(df_network, live_matrix_match, on="Ticker", how="inner")
                st.dataframe(final_mapped_network_df, use_container_width=True, hide_index=True)
            else:
                st.warning("To map Whale records, run a scan with tickers: NVDA, MSFT, CEG, PHYS, CYBR, GNK, AAPL, OXY, BAC.")
        else:
            st.info("Execute scanner velocity sweeps to run data overlay matching.")

    with tab6:
        st.header("🔥 Aggressive Momentum Playground")
        st.info("📅 Top 10 Daily Momentum Stocks - Auto-selected by volume velocity & momentum metrics. Refreshes daily.")
        st.warning("⚠️ RISK NOTICE: This workspace runs a short 20-day historical data gate with aggressive thresholds. Highly volatile assets may report false breakout signals. Use tighter stops.")
        if not st.session_state.new_swings_results.empty:
            exclude_swings = ["Chg_4W_Raw", "Ratio_52W_Raw", "Zacks_Rank", "EPS_Revision_Delta", "Operating_Margin", "ROA", "Score_Internal_Num"]
            display_swings_cols = [c for c in st.session_state.new_swings_results.columns if c not in exclude_swings]
            st.dataframe(st.session_state.new_swings_results[display_swings_cols], use_container_width=True, hide_index=True)
            
            # Add visualization for top momentum picks
            if not st.session_state.new_swings_results.empty:
                st.subheader("📊 Momentum Score Distribution")
                momentum_viz_data = st.session_state.new_swings_results[['Ticker', 'xVOL Velocity']].copy()
                momentum_viz_data['xVOL_Numeric'] = momentum_viz_data['xVOL Velocity'].astype(str).str.replace('x', '', regex=False).astype(float)
                fig_momentum = go.Figure(data=[
                    go.Bar(x=momentum_viz_data['Ticker'], y=momentum_viz_data['xVOL_Numeric'], marker_color='#FF6B35')
                ])
                fig_momentum.update_layout(template="plotly_dark", title="Volume Velocity Ranking", xaxis_title="Ticker", yaxis_title="xVOL Multiplier", height=400)
                st.plotly_chart(fig_momentum, use_container_width=True)
        else:
            st.info("⏳ Loading auto-selected top 10 momentum stocks...")
