    with tab2:
        valid_selections = [t for t in t_list if t in st.session_state.bulk_data]
        if valid_selections:
            sel = st.radio("Asset Pivot View:", valid_selections, horizontal=True)
            if sel and sel in st.session_state.bulk_data:
                df_plot = st.session_state.bulk_data[sel].copy()
                df_plot.index = pd.to_datetime(df_plot.index)
               
                # FIXED: Corrected dictionary syntax declaration mapping for the multi-axis canvas grid
                fig = make_subplots(specs=[[{"secondary_y": True}]])
               
                # Plot price series components on primary y-axis
                fig.add_trace(go.Candlestick(x=df_plot.index, open=df_plot['Open'], high=df_plot['High'], low=df_plot['Low'], close=df_plot['Close'], name="Price"), secondary_y=False)
                if 'SMA200' in df_plot.columns:
                    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['SMA200'], line=dict(color='gold', width=2), name='SMA 200'), secondary_y=False)
                if 'SMA50' in df_plot.columns:
                    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['SMA50'], line=dict(color='cyan', width=1), name='SMA 50'), secondary_y=False)
               
                # Plot continuous volume trace on secondary y-axis
                fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['Volume'], line=dict(color='rgba(255, 165, 0, 0.6)', width=1.5), name='Volume Tracking'), secondary_y=True)
               
                fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=600, margin=dict(t=20, b=20, l=20, r=20))
                fig.update_yaxes(title_text="<b>Stock Share Price ($)</b>", secondary_y=False)
                fig.update_yaxes(title_text="<b>Institutional Liquidity Volume Curve</b>", secondary_y=True)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Execute scan sweeps to display charting visuals.")
