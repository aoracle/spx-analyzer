import streamlit as st
import pandas as pd
import sys
from spx_5min import SPX5MinCandleAnalyzer
from datetime import datetime

st.set_page_config(page_title="SPX 5-Minute Analyzer", layout="wide")
st.title("SPX 5-Minute Candle Analyzer Dashboard")

# Sidebar for options
date_str = st.sidebar.text_input("Enter date (YYYYMMDD)", value=datetime.now().strftime('%Y%m%d'))
debug = st.sidebar.checkbox("Debug mode", value=False)
mode = st.sidebar.selectbox("Mode", ["historical", "quick_view", "simulation"])

analyzer = SPX5MinCandleAnalyzer(debug=debug)

run_analysis = st.sidebar.button("Run Analysis")

if run_analysis:
    with st.spinner("Running analysis..."):
        try:
            analyzer.load_historical_data(date_str)
            if mode == "historical":
                analyzer.update_all_levels(date_str)
                target_data = analyzer.historical_data[analyzer.historical_data['date'] == date_str].reset_index(drop=True)
                if not target_data.empty:
                    st.subheader(f"SPX Historical Analysis - {date_str}")
                    st.write(f"Total candles: {len(target_data)}")
                    current_price = target_data.iloc[-1]['close']
                    
                    # Display static and technical levels
                    st.write("### Static Price Levels (High to Low):")
                    from tabulate import tabulate
                    static_levels = [
                        (k, v) for k, v in analyzer.levels.items()
                        if k in ['PPWH', 'PMH', 'PWH', 'PPWL', 'MH', 'ORBH', 'PDH', 'PDC', 'ORBL', 'PDL', 'P2DH', 'P2DL', 'P3DH', 'P3DL', 'PWL', 'PML', 'ML']
                    ]
                    static_levels.append(('CURRENT', {'desc': 'Current Price', 'value': current_price}))
                    static_levels.sort(key=lambda x: x[1]['value'], reverse=True)
                    
                    # Create DataFrame for static levels
                    static_data = []
                    for code, info in static_levels:
                        if info['value'] > 0 or code == 'CURRENT':
                            distance = current_price - info['value']
                            static_data.append({
                                'Level': code,
                                'Description': info['desc'],
                                'Price': f"${info['value']:.2f}",
                                'Distance': f"{distance:+.2f}"
                            })
                    st.dataframe(pd.DataFrame(static_data))

                    # Dynamic Levels
                    st.write("### Dynamic Levels:")
                    dynamic_data = []
                    for span in [8, 17, 24]:
                        ema_value = analyzer.levels[f'EMA{span}']['value']
                        if ema_value > 0:
                            distance = current_price - ema_value
                            dynamic_data.append({
                                'Indicator': f"EMA{span}",
                                'Value': f"${ema_value:.2f}",
                                'Distance': f"{abs(distance):.2f}",
                                'Position': "Above" if current_price > ema_value else "Below"
                            })
                    for ott_type in ['HOTT', 'LOTT']:
                        if ott_type in analyzer.levels and analyzer.levels[ott_type]['value'] > 0:
                            ott_value = analyzer.levels[ott_type]['value']
                            distance = current_price - ott_value
                            dynamic_data.append({
                                'Indicator': f"{ott_type} (2,1.5,22)",
                                'Value': f"${ott_value:.2f}",
                                'Distance': f"{abs(distance):.2f}",
                                'Position': "Above" if current_price > ott_value else "Below"
                            })
                    st.dataframe(pd.DataFrame(dynamic_data))

                    st.write("### Candle Data Table")
                    try:
                        analysis_df = analyzer.get_candle_analysis_table(target_data)
                        if not analysis_df.empty:
                            # Clean and prepare DataFrame
                            analysis_df = analysis_df.copy()
                            analysis_df = analysis_df.fillna("")
                            
                            # Convert timestamp to readable format
                            if 'timestamp' in analysis_df.columns:
                                analysis_df['Time'] = pd.to_datetime(analysis_df['timestamp']).dt.strftime('%H:%M')
                                analysis_df = analysis_df.drop(columns=['timestamp'])
                            
                            # Calculate Body column if not present
                            if 'Body' not in analysis_df.columns:
                                analysis_df['Body'] = analysis_df['close'] - analysis_df['open']
                            
                            # Add Trend column
                            analysis_df['Trend'] = analysis_df.apply(
                                lambda row: analyzer.get_trend_label(
                                    row['ema8'] if 'ema8' in row else 0,
                                    row['ema21'] if 'ema21' in row else 0,
                                    row['ema55'] if 'ema55' in row else 0
                                ),
                                axis=1
                            )
                            
                            # Add Price vs EMAs column
                            analysis_df['Price vs EMAs'] = analysis_df.apply(
                                lambda row: "Above All" if row['close'] > row['ema8'] and row['close'] > row['ema21'] and row['close'] > row['ema55']
                                else "Below All" if row['close'] < row['ema8'] and row['close'] < row['ema21'] and row['close'] < row['ema55']
                                else "Mixed",
                                axis=1
                            )
                            
                            # Add Important Levels column
                            def get_important_levels(row):
                                current_price = row['close']
                                static_levels = [(k, v['value']) for k, v in analyzer.levels.items()
                                               if k in ['PPWH', 'PMH', 'PWH', 'PPWL', 'MH', 'ORBH', 'PDH', 'PP', 'PDC', 'ORBL', 'PDL', 'P2DH', 'P2DL', 'P3DH', 'P3DL']]
                                
                                levels_above = [level for level in static_levels if level[1] > current_price]
                                nearest_above = min(levels_above, key=lambda x: x[1] - current_price) if levels_above else ('None', current_price)
                                levels_below = [level for level in static_levels if level[1] < current_price]
                                nearest_below = max(levels_below, key=lambda x: x[1] - current_price) if levels_below else ('None', current_price)
                                
                                important_levels = []
                                if nearest_above[0] != 'None' and nearest_above[1] - current_price <= 5:
                                    important_levels.append(f"{nearest_above[0]}(+{nearest_above[1] - current_price:.2f})")
                                if nearest_below[0] != 'None' and current_price - nearest_below[1] <= 5:
                                    important_levels.append(f"{nearest_below[0]}(-{current_price - nearest_below[1]:.2f})")
                                if analyzer.is_psychological_level(current_price):
                                    important_levels.append("PSYCH")
                                
                                return ", ".join(important_levels) if important_levels else "-"
                            
                            analysis_df['Important Levels'] = analysis_df.apply(get_important_levels, axis=1)
                            
                            # Add HTF columns
                            analysis_df['HTF_Impulse_Trend'] = analyzer.add_htf_impulse_trend(target_data)
                            analysis_df['HTF_Sweep_Flag'] = analyzer.add_htf_sweep_flag(target_data)
                            analysis_df['HTF_Zone_Tag'] = analyzer.add_htf_zone_tag(target_data)
                            
                            # Rename columns for display
                            column_mapping = {
                                'open': 'Open',
                                'high': 'High',
                                'low': 'Low',
                                'close': 'Close',
                                'shendi': 'Upper Shadow',
                                'shepti': 'Lower Shadow',
                                'ema8': 'EMA8',
                                'ema21': 'EMA21',
                                'ema55': 'EMA55'
                            }
                            analysis_df = analysis_df.rename(columns=column_mapping)
                            
                            # Format numeric columns
                            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Body', 'Upper Shadow', 'Lower Shadow', 'EMA8', 'EMA21', 'EMA55']
                            for col in numeric_columns:
                                if col in analysis_df.columns:
                                    analysis_df[col] = analysis_df[col].apply(lambda x: f"{float(x):.2f}" if pd.notnull(x) else "")
                            
                            # Limit to last 100 rows
                            analysis_df = analysis_df.tail(100)
                            
                            # Apply styling
                            def style_dataframe(df):
                                def style_cell(x):
                                    try:
                                        if isinstance(x, str):
                                            if x == '-':
                                                return ''
                                            if x.startswith('-'):
                                                try:
                                                    val = float(x)
                                                    return 'color: red' if val < 0 else 'color: green'
                                                except ValueError:
                                                    return ''
                                        elif isinstance(x, (int, float)):
                                            return 'color: green' if x > 0 else 'color: red' if x < 0 else ''
                                    except:
                                        pass
                                    return ''
                                
                                return df.style.applymap(style_cell)
                            
                            st.dataframe(style_dataframe(analysis_df))
                        else:
                            st.warning("No analysis data available")
                    except Exception as e:
                        st.error(f"Error displaying candle data: {str(e)}")
                        if debug:
                            st.exception(e)
                else:
                    st.error(f"No data found for date {date_str}")
            elif mode == "quick_view":
                analyzer.update_all_levels(date_str)
                target_data = analyzer.historical_data[analyzer.historical_data['date'] == date_str].reset_index(drop=True)
                if not target_data.empty:
                    st.subheader(f"SPX Quick View Analysis - {date_str}")
                    st.write("### Last 10 Candles")
                    styled_df = analyzer.get_candle_analysis_styled_table(target_data.tail(10))
                    st.write(styled_df.to_html(), unsafe_allow_html=True)
                else:
                    st.error(f"No data found for date {date_str}")
            elif mode == "simulation":
                analyzer.update_all_levels(date_str)
                target_data = analyzer.historical_data[analyzer.historical_data['date'] == date_str].reset_index(drop=True)
                if not target_data.empty:
                    st.subheader(f"SPX Simulation - {date_str}")
                    candle_index = st.slider("Candle Index", 1, len(target_data), 1)
                    styled_df = analyzer.get_candle_analysis_styled_table(target_data.iloc[:candle_index])
                    st.write(styled_df.to_html(), unsafe_allow_html=True)
                else:
                    st.error(f"No data found for date {date_str}")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            if debug:
                st.exception(e)


