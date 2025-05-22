"""
SPX 5-Minute Candle Analyzer
Analyzes price action candle by candle with static and dynamic levels
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from tabulate import tabulate

from colorama import init, Fore, Style
from pathlib import Path
import time
import traceback
import re
import pandas as pd
from datetime import datetime
import os
from scipy.stats import linregress
from scipy.signal import argrelextrema
from scipy.signal import savgol_filter
import sys
import pdb
import requests
import streamlit as st

# Initialize colorama
init()

class SPX5MinCandleAnalyzer:
    def __init__(self, debug=False):
        self.debug = debug
        self.est = pytz.timezone('US/Eastern')
        self.data_dir = Path("/Users/anand/Dropbox/Working_scripts/historical_data")
        
        # Initialize data structures
        self.historical_data = pd.DataFrame()
        self.today_data = pd.DataFrame()
        
        # Store EMAs for each candle
        self.ema_history = {
            8: {},   # Will store {timestamp: value} for each candle
            17: {},
            24: {}
        }
        
        # Initialize levels dictionary with all required keys
        self.levels = {
            'PP': {'desc': "Pivot Point", 'value': 0},
            'PDH': {'desc': "Prior Day High", 'value': 0},
            'PDL': {'desc': "Prior Day Low", 'value': 0},
            'PDC': {'desc': "Prior Day Close", 'value': 0},
            'ORBH': {'desc': "Opening Range High", 'value': 0},
            'ORBL': {'desc': "Opening Range Low", 'value': 0},
            'EMA8': {'desc': "8 EMA", 'value': 0},
            'EMA17': {'desc': "17 EMA", 'value': 0},
            'EMA24': {'desc': "24 EMA", 'value': 0},
            'HOTT': {'desc': "High OTT", 'value': 0},
            'LOTT': {'desc': "Low OTT", 'value': 0},
            'PPWH': {'desc': "Prior Prior Week High", 'value': 0},
            'PWH': {'desc': "Prior Week High", 'value': 0},
            'PPWL': {'desc': "Prior Prior Week Low", 'value': 0},
            'PWL': {'desc': "Prior Week Low", 'value': 0},
            'PMH': {'desc': "Prior Month High", 'value': 0},
            'PML': {'desc': "Prior Month Low", 'value': 0},
            'MH': {'desc': "Month High", 'value': 0},
            'ML': {'desc': "Month Low", 'value': 0},
            'WH': {'desc': "Week High", 'value': 0},
            'WL': {'desc': "Week Low", 'value': 0},
            'P2DH': {'desc': 'Previous 2 Day High', 'value': 0},
            'P2DL': {'desc': 'Previous 2 Day Low', 'value': 0},
            'P3DH': {'desc': 'Previous 3 Day High', 'value': 0},
            'P3DL': {'desc': 'Previous 3 Day Low', 'value': 0}
        }

        # Add pattern detection parameters
        self.pattern_params = {
            'channel_lookback': 20,
            'channel_slope_tolerance': 0.001,
            'top_bottom_lookback': 20,
            'top_bottom_price_tolerance': 0.5,
            'triangle_lookback': 20,
            'triangle_tolerance': 0.5,
            'breakout_lookback': 20,
            'breakout_std_multiplier': 2.0
        }
        
        # Pattern abbreviation mapping
        self.pattern_abbrev = {
            'ASCENDING TRIANGLE': ('AT', True),    # (abbreviation, is_bullish)
            'DOWNSIDE BREAKOUT': ('DB', False),
            'UPSIDE BREAKOUT': ('UB', True),
            'INVERSE HEAD AND SHOULDERS': ('IHS', True),
            'HEAD AND SHOULDERS': ('H&S', False),
            'DOWN CHANNEL': ('DC', False),
            'UP CHANNEL': ('UC', True),
            'DOUBLE TOP': ('DT', False),
            'DOUBLE BOTTOM': ('DB', True),
            'TRIPLE TOP': ('TT', False),
            'TRIPLE BOTTOM': ('TB', True)
        }

    def fetch_historical_ohlc_data(self, root, start_date, end_date, interval_ms, api_key):
        """Fetches historical OHLC data from Thetadata REST API."""
        base_url = "https://api.thetadata.us/v2/indices/ohlc"
        params = {
            'root': root,
            'start_date': start_date,
            'end_date': end_date,
            'ivl': interval_ms,
            'api_key': api_key,
            'use_csv': 'false', # Requesting JSON for easier parsing
            'pretty_time': 'false'
        }
        
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status() # Raise an exception for bad status codes
            data = response.json()
            
            if data and data['response']:
                # Extract data and format
                ohlc_data = data['response']
                columns = data['header']['format'] # Get column names from header
                df = pd.DataFrame(ohlc_data, columns=columns)
                
                # Convert ms_of_day and date to timestamp
                # ms_of_day is milliseconds since midnight ET
                # date is YYYYMMDD
                df['timestamp'] = pd.to_datetime(df['date'].astype(str), format='%Y%m%d') + pd.to_timedelta(df['ms_of_day'], unit='ms')
                
                # Drop original date and ms_of_day columns
                df = df.drop(columns=['ms_of_day', 'date'])
                
                return df
            else:
                print(f"{Fore.YELLOW}Thetadata API returned no data for {root} from {start_date} to {end_date}{Style.RESET_ALL}")
                return pd.DataFrame()
                
        except requests.exceptions.RequestException as e:
            print(f"{Fore.RED}Error fetching data from Thetadata API: {e}{Style.RESET_ALL}")
            if self.debug:
                traceback.print_exc()
            return pd.DataFrame()

    def load_historical_data(self, target_date=None):
        """Load and process historical data"""
        try:
            # Check if running on Streamlit Cloud (environment variable set by Streamlit)
            is_streamlit_cloud = os.environ.get('STREAMLIT_CLOUD', 'false').lower() == 'true'

            if is_streamlit_cloud:
                if 'THETADATA_API_KEY' not in st.secrets:
                    st.error("Thetadata API key not found in Streamlit secrets.")
                    print(f"{Fore.RED}Thetadata API key not found in Streamlit secrets.{Style.RESET_ALL}")
                    return
                    
                api_key = st.secrets['THETADATA_API_KEY']
                root_symbol = 'SPX' # Or whatever the correct root is for SPX indices
                # Determine the date range to fetch
                # For historical view, let's fetch the last 40 days for now, similar to local loading
                today = datetime.now(self.est)
                if target_date:
                     today = datetime.strptime(target_date, '%Y%m%d').replace(tzinfo=self.est)

                # Fetch data for the last 40 days
                all_data = []
                for i in range(40):
                    check_date = today - timedelta(days=i)
                    date_str = check_date.strftime('%Y%m%d')
                    
                    # ThetaData API might return data even for weekends if available, fetch day by day
                    day_df = self.fetch_historical_ohlc_data(
                        root=root_symbol,
                        start_date=date_str,
                        end_date=date_str,
                        interval_ms=300000, # 5 minutes
                        api_key=api_key
                    )
                    if not day_df.empty:
                        all_data.append(day_df)
                
                if all_data:
                    self.historical_data = pd.concat(all_data, ignore_index=True)
                    self.historical_data = self.historical_data.sort_values('timestamp').reset_index(drop=True)
                    # Resample is not needed if API returns 5-minute intervals directly
                    print(f"{Fore.GREEN}Loaded historical data from Thetadata REST API successfully{Style.RESET_ALL}")
                else:
                    print(f"{Fore.RED}No historical data found from Thetadata REST API{Style.RESET_ALL}")
                
            else:
                # Existing local file loading logic
                data = []
                
                # Convert month number to month name
                month_map = {
                    '01': 'JAN', '02': 'FEB', '03': 'MAR', '04': 'APR',
                    '05': 'MAY', '06': 'JUN', '07': 'JUL', '08': 'AUG',
                    '09': 'SEP', '10': 'OCT', '11': 'NOV', '12': 'DEC'
                }
                
                # Load last 40 trading days to ensure we get previous month data
                today = datetime.now(self.est)
                if target_date:
                    # If target_date is provided, use it as the reference date
                    today = datetime.strptime(target_date, '%Y%m%d').replace(tzinfo=self.est)

                for i in range(40):
                    check_date = today - timedelta(days=i)
                    if check_date.weekday() >= 5:  # Skip weekends
                        continue
                        
                    date_str = check_date.strftime('%Y%m%d')
                    year = date_str[:4]
                    month_num = date_str[4:6]
                    month = month_map[month_num]
                    
                    file_path = self.data_dir / year / month / f"spx_{date_str[6:8]}{month_num}{date_str[2:4]}.csv"
                    
                    if file_path.exists():
                        try:
                            # Assuming local files are already in 5-minute format
                            day_data = pd.read_csv(file_path)
                            day_data = day_data[
                                (day_data['open'] > 0) &
                                (day_data['high'] > 0) &
                                (day_data['low'] > 0) &
                                (day_data['close'] > 0)
                            ]
                            day_data['date'] = date_str
                            data.append(day_data)
                            if self.debug:
                                print(f"Loaded data from {file_path}")
                        except Exception as e:
                            if self.debug:
                                print(f"Error loading {file_path}: {str(e)}")
                            continue
                
                if data:
                    self.historical_data = pd.concat(data, ignore_index=True)
                    self.historical_data['timestamp'] = pd.to_datetime(self.historical_data['timestamp'])
                    # Sort and reset index
                    self.historical_data = self.historical_data.sort_values('timestamp').reset_index(drop=True)
                    
                    print(f"{Fore.GREEN}Loaded historical data from local files successfully{Style.RESET_ALL}")
                else:
                    print(f"{Fore.RED}No historical data found from local files{Style.RESET_ALL}")
                
        except Exception as e:
            print(f"{Fore.RED}Error loading historical data: {e}{Style.RESET_ALL}")
            if self.debug:
                traceback.print_exc()

    def get_dynamic_level_status(self, current_price, timestamp=None):
        """Get status relative to dynamic levels (EMAs)"""
        # Get EMA values
        ema_levels = {
            'EMA8': self.get_ema_for_candle(timestamp, 8) or 0,
            'EMA17': self.get_ema_for_candle(timestamp, 17) or 0,
            'EMA24': self.get_ema_for_candle(timestamp, 24) or 0   
        }
        
        valid_emas = [(k, v) for k, v in ema_levels.items() if v > 0]
        if not valid_emas:
            return "N/A"
        
        # Find highest and lowest EMAs
        highest_ema = max(valid_emas, key=lambda x: x[1])
        lowest_ema = min(valid_emas, key=lambda x: x[1])
        
        # Determine trend based on EMA alignment
        is_downtrend = ema_levels['EMA8'] < ema_levels['EMA17'] < ema_levels['EMA24']
        is_uptrend = ema_levels['EMA8'] > ema_levels['EMA17'] > ema_levels['EMA24']
        
        # Get the current candle's high, low, and close
        current_candle = self.historical_data[self.historical_data['timestamp'] == timestamp].iloc[0]
        candle_close = current_candle['close']
        
        # Check for reversal conditions with full candle close
        if is_downtrend and candle_close > highest_ema[1]:
            return f"{Fore.GREEN}Above {highest_ema[0]} ↑{Style.RESET_ALL}"
        elif is_uptrend and candle_close < lowest_ema[1]:
            return f"{Fore.RED}Below {lowest_ema[0]} ↓{Style.RESET_ALL}"
        else:
            return "Neutral"

    def write_candle_analysis_to_file(self, data, date_str):
        """Write candle analysis data to a CSV file with timestamp."""
        try:
            output_dir = "candle_analysis_data"
            os.makedirs(output_dir, exist_ok=True)
            filename = f"{output_dir}/candle_analysis_{date_str}.csv"
            
            # Remove ANSI color codes from the data
            clean_data = []
            for row in data:
                clean_row = [self.remove_color_codes(item) for item in row]
                clean_data.append(clean_row)
            
            # Create a DataFrame with all columns including new ones
            df = pd.DataFrame(clean_data, columns=[
                'Time', 'NO', 'SAbove', 'DIS', 'Close', 'Body',
                'SBelow', 'DIS', 'SRange', 'DLevel', 'EMA', 'EMAST',
                'DayHigh', 'DayLow' ,'Impulse_Trend', 'Sweep_Flag', 'Zone_Tag', 'PP', 'SP'
            ])
            
            # Write the DataFrame to a CSV file
            df.to_csv(filename, index=False)
            print(f"Data written to {filename}")
        except Exception as e:
            print(f"Error writing candle analysis to file: {e}")

    def remove_color_codes(self, text):
        """Remove ANSI color codes from text"""
        return re.sub(r'\x1b\[[0-9;]*m', '', str(text))

    def is_psychological_level(self, price, threshold=3):
        """Check if price is near a psychological level"""
        # Define SPX psychological levels
        psych_levels = [5400,5450,
            5500, 5550, 5600, 5650, 5700, 5750, 5800, 5850, 
            5900, 5950, 6000, 6050, 6100
        ]
        
        # Check if price is within threshold points of any psychological level
        for level in psych_levels:
            if abs(price - level) <= threshold:
                return True
        return False

    def get_trend_label(self, ema8, ema21, ema55, threshold=0.1):
        """
        Determine trend label based on EMA alignment and proximity.
        Uses 8-day, 21-day, and 55-day EMAs.
        Returns 'Above EMA55 ↑', 'Below EMA55 ↓', or 'Neutral'.
        """
        if ema8 > ema21 > ema55:
            return f"{Fore.GREEN}Above EMA55 ↑{Style.RESET_ALL}"
        elif ema8 < ema21 < ema55:
            return f"{Fore.RED}Below EMA55 ↓{Style.RESET_ALL}"
        else:
            return "Neutral"

    def display_candle_analysis(self, data):
        """Display candle by candle analysis"""
        data = data.copy()

        if 'shendi' not in data.columns:
            data['shendi'] = data.apply(lambda row: row['high'] - max(row['open'], row['close']), axis=1)
        if 'shepti' not in data.columns:
            data['shepti'] = data.apply(lambda row: min(row['open'], row['close']) - row['low'], axis=1)
        if 'ema8' not in data.columns:
            data['ema8'] = data['close'].ewm(span=8, adjust=False).mean()
        if 'ema21' not in data.columns:
            data['ema21'] = data['close'].ewm(span=21, adjust=False).mean()
        if 'ema55' not in data.columns:
            data['ema55'] = data['close'].ewm(span=55, adjust=False).mean()
        if 'price_vs_ema' not in data.columns:
            def price_vs_ema_fn(row):
                if row['close'] > row['ema8'] and row['close'] > row['ema21'] and row['close'] > row['ema55']:
                    return f"{Fore.GREEN}Above All{Style.RESET_ALL}"
                elif row['close'] < row['ema8'] and row['close'] < row['ema21'] and row['close'] < row['ema55']:
                    return f"{Fore.RED}Below All{Style.RESET_ALL}"
                else:
                    return f"{Fore.YELLOW}Mixed{Style.RESET_ALL}"
            data['price_vs_ema'] = data.apply(price_vs_ema_fn, axis=1)

        display_rows = []
        current_day_high = float('-inf')
        current_day_low = float('inf')
        data['HTF_Impulse_Trend'] = self.add_htf_impulse_trend(data)
        data['HTF_Sweep_Flag'] = self.add_htf_sweep_flag(data)
        data['HTF_Zone_Tag'] = self.add_htf_zone_tag(data)

        max_body = data['close'] - data['open']
        max_body_val = max_body.max()
        min_body_val = max_body.min()
        max_shendi_val = data['shendi'].max()
        max_shepti_val = data['shepti'].max()

        for idx, row in data.iterrows():
            candle_num = idx + 1
            data_until_now = data.iloc[:idx+1]
            day_high = data_until_now['high'].max()
            day_low = data_until_now['low'].min()
            is_new_high = day_high > current_day_high
            is_new_low = day_low < current_day_low
            current_day_high = max(current_day_high, day_high)
            current_day_low = min(current_day_low, day_low)

            current_price = row['close']
            time_str = pd.to_datetime(row['timestamp']).strftime('%H:%M')
            candle_time = pd.to_datetime(row['timestamp']).time()
            or_active_time = datetime.strptime("10:15", "%H:%M").time()

            body_length = row['close'] - row['open']
            body_color = Fore.GREEN if body_length >= 0 else Fore.RED

            highlight_row = False
            if body_length == max_body_val or body_length == min_body_val:
                highlight_row = True

            open_val = f"{row['open']:.2f}"
            high_val = f"{row['high']:.2f}"
            low_val = f"{row['low']:.2f}"
            close_val = f"{row['close']:.2f}"

            if row['high'] == day_high:
                high_val = f"{Fore.GREEN}{high_val}{Style.RESET_ALL}"
            if row['low'] == day_low:
                low_val = f"{Fore.RED}{low_val}{Style.RESET_ALL}"

            if row['open'] == day_high:
                open_val = f"{Fore.GREEN}{open_val}{Style.RESET_ALL}"
            elif row['open'] == day_low:
                open_val = f"{Fore.RED}{open_val}{Style.RESET_ALL}"
            if row['close'] == day_high:
                close_val = f"{Fore.GREEN}{close_val}{Style.RESET_ALL}"
            elif row['close'] == day_low:
                close_val = f"{Fore.RED}{close_val}{Style.RESET_ALL}"

            shendi_val = f"{row['shendi']:.2f}"
            shepti_val = f"{row['shepti']:.2f}"
            if row['shendi'] == max_shendi_val:
                shendi_val = f"{Fore.MAGENTA}{shendi_val}{Style.RESET_ALL}"
            if row['shepti'] == max_shepti_val:
                shepti_val = f"{Fore.MAGENTA}{shepti_val}{Style.RESET_ALL}"

            static_levels = [(k, v['value']) for k, v in self.levels.items()
                            if k in ['PPWH', 'PMH', 'PWH', 'PPWL', 'MH', 'ORBH',
                                   'PDH', 'PP', 'PDC', 'ORBL', 'PDL', 'P2DH', 'P2DL', 'P3DH', 'P3DL']]
            if candle_time < or_active_time:
                static_levels = [(k, v) for k, v in static_levels if k not in ['ORBH', 'ORBL']]
            levels_above = [level for level in static_levels if level[1] > current_price]
            nearest_above = min(levels_above, key=lambda x: x[1] - current_price) if levels_above else ('None', current_price)
            levels_below = [level for level in static_levels if level[1] < current_price]
            nearest_below = max(levels_below, key=lambda x: x[1] - current_price) if levels_below else ('None', current_price)
            distance_to_above = nearest_above[1] - current_price if nearest_above[0] != 'None' else 0
            distance_to_below = current_price - nearest_below[1] if nearest_below[0] != 'None' else 0
            important_levels = []
            if nearest_above[0] != 'None' and distance_to_above <= 5:
                important_levels.append(f"{nearest_above[0]}(+{distance_to_above:.2f})")
            if nearest_below[0] != 'None' and distance_to_below <= 5:
                important_levels.append(f"{nearest_below[0]}(-{distance_to_below:.2f})")
            if self.is_psychological_level(current_price):
                important_levels.append("PSYCH")
            important_levels_str = ", ".join(important_levels) if important_levels else "-"

            row_items = [
                time_str,
                open_val,
                high_val,
                low_val,
                close_val,
                f"{body_color}{body_length:+.2f}{Style.RESET_ALL}",
                shendi_val,
                shepti_val,
                f"{row['ema8']:.2f}",
                f"{row['ema21']:.2f}",
                f"{row['ema55']:.2f}",
                self.get_trend_label(row['ema8'], row['ema21'], row['ema55']),
                row['price_vs_ema'],
                important_levels_str
            ]

            display_rows.append(row_items)
        headers = [
            "Time", "Open", "High", "Low", "Close", "Body",
            "Shendi", "Shepti", "EMA8", "EMA21", "EMA55",
            "Trend", "Price vs EMAs", "Important Levels"
        ]
        print(tabulate(display_rows, headers=headers, tablefmt="fancy_grid"))

    def calculate_ema_on_2min(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate EMAs on 2-minute candles"""
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data_2min = data.resample('2T', on='timestamp').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }).reset_index()

        for span in [8, 17, 24]:
            ema_series = data_2min['close'].ewm(span=span, adjust=False).mean()
            data_2min[f'ema{span}'] = ema_series

        data = data.merge(data_2min[['timestamp', 'ema8', 'ema17', 'ema24']],
                         on='timestamp', how='left')

        return data

    def convert_5min_to_1min(self, data: pd.DataFrame) -> pd.DataFrame:
        """Convert 5-minute data to 1-minute data with EMAs"""
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data_1min = data.resample('1T', on='timestamp').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }).reset_index()

        data_1min = self.calculate_ema_on_2min(data_1min)

        return data_1min

    def calculate_ott(self, data):
        """Calculate OTT (Optimized Trend Tracker)"""
        if len(data) < 2:
            return pd.Series(0, index=data.index), pd.Series(0, index=data.index)

        var = data['close'].ewm(span=22, adjust=False).var()
        std = np.sqrt(var)

        ma = data['close'].ewm(span=2, adjust=False).mean()

        k = 1.5 / 100

        hott = ma + std * k
        lott = ma - std * k

        return hott, lott

    def update_weekly_levels(self):
        """Update weekly high and low levels"""
        if self.historical_data.empty:
            return

        self.historical_data['datetime'] = pd.to_datetime(self.historical_data['date'])

        latest_date = self.historical_data['datetime'].max()

        if self.debug:
            print("\nDEBUG - Weekly Levels:")
            print(f"Latest date: {latest_date}")

        current_week_start = latest_date - timedelta(days=latest_date.weekday())
        prev_week_start = current_week_start - timedelta(days=7)
        prev_prev_week_start = prev_week_start - timedelta(days=7)

        if self.debug:
            print(f"Current week start: {current_week_start}")
            print(f"Previous week start: {prev_week_start}")
            print(f"Prior previous week start: {prev_prev_week_start}")

        current_week_data = self.historical_data[
            (self.historical_data['datetime'] >= current_week_start) &
            (self.historical_data['datetime'] <= latest_date)
        ]
        prev_week_data = self.historical_data[
            (self.historical_data['datetime'] >= prev_week_start) &
            (self.historical_data['datetime'] < current_week_start)
        ]
        prev_prev_week_data = self.historical_data[
            (self.historical_data['datetime'] >= prev_prev_week_start) &
            (self.historical_data['datetime'] < prev_week_start)
        ]

        if self.debug:
            print("\nData points found:")
            print(f"Current week: {len(current_week_data)} candles")
            print(f"Previous week: {len(prev_week_data)} candles")
            print(f"Prior previous week: {len(prev_prev_week_data)} candles")

        if not current_week_data.empty:
            self.levels['WH']['value'] = float(current_week_data['high'].max())
            self.levels['WL']['value'] = float(current_week_data['low'].min())
            if self.debug:
                print(f"\nCurrent Week Levels - WH: {self.levels['WH']['value']:.2f}, WL: {self.levels['WL']['value']:.2f}")

        if not prev_week_data.empty:
            self.levels['PWH']['value'] = float(prev_week_data['high'].max())
            self.levels['PWL']['value'] = float(prev_week_data['low'].min())
            if self.debug:
                print(f"Previous Week Levels - PWH: {self.levels['PWH']['value']:.2f}, PWL: {self.levels['PWL']['value']:.2f}")

        if not prev_prev_week_data.empty:
            self.levels['PPWH']['value'] = float(prev_prev_week_data['high'].max())
            self.levels['PPWL']['value'] = float(prev_prev_week_data['low'].min())
            if self.debug:
                print(f"Prior Previous Week Levels - PPWH: {self.levels['PPWH']['value']:.2f}, PPWL: {self.levels['PPWL']['value']:.2f}")

        self.historical_data.drop('datetime', axis=1, inplace=True)

    def calculate_pivot_point(self, high, low, close):
        """Calculate Daily Pivot Point"""
        pivot = (high + low + close) / 3
        self.levels['PP']['value'] = float(pivot)

    def update_monthly_levels(self, target_date=None):
        """Update monthly high and low levels"""
        if self.historical_data.empty:
            return

        self.historical_data['datetime'] = pd.to_datetime(self.historical_data['date'])
        reference_date = pd.to_datetime(target_date) if target_date else self.historical_data['datetime'].max()

        if self.debug:
            print("\nDEBUG - Monthly Levels:")
            print(f"Reference date: {reference_date}")

        current_month_start = reference_date.replace(day=1)
        prev_month_start = (current_month_start - timedelta(days=1)).replace(day=1)

        if self.debug:
            print(f"Current month start: {current_month_start}")
            print(f"Previous month start: {prev_month_start}")

        current_month_data = self.historical_data[
            (self.historical_data['datetime'] >= current_month_start) &
            (self.historical_data['datetime'] <= reference_date)
        ]

        prev_month_data = self.historical_data[
            (self.historical_data['datetime'] >= prev_month_start) &
            (self.historical_data['datetime'] < current_month_start)
        ]

        if self.debug:
            print("\nData points found:")
            print(f"Current month: {len(current_month_data)} candles")
            print(f"Previous month: {len(prev_month_data)} candles")

        if not current_month_data.empty:
            self.levels['MH']['value'] = float(current_month_data['high'].max())
            self.levels['ML']['value'] = float(current_month_data['low'].min())
            if self.debug:
                print(f"\nCurrent Month Levels - MH: {self.levels['MH']['value']:.2f}, ML: {self.levels['ML']['value']:.2f}")

        if not prev_month_data.empty:
            self.levels['PMH']['value'] = float(prev_month_data['high'].max())
            self.levels['PML']['value'] = float(prev_month_data['low'].min())
            if self.debug:
                print(f"Previous Month Levels - PMH: {self.levels['PMH']['value']:.2f}, PML: {self.levels['PML']['value']:.2f}")

        self.historical_data.drop('datetime', axis=1, inplace=True)

    def update_all_levels(self, target_date=None, current_candle_idx=None):
        """Update all price levels from historical data"""
        if self.historical_data.empty:
            return

        reference_date = target_date if target_date else self.historical_data['date'].max()
        today_data = self.historical_data[self.historical_data['date'] == reference_date].copy()

        if current_candle_idx is not None:
            today_data = today_data.iloc[:current_candle_idx + 1]

        prev_dates = pd.Series(
            self.historical_data[self.historical_data['date'] < reference_date]['date'].unique()
        ).sort_values(ascending=False).iloc[:3]

        if len(prev_dates) >= 1:
            prev_data = self.historical_data[self.historical_data['date'] == prev_dates.iloc[0]]
            self.levels['PDH']['value'] = float(prev_data['high'].max())
            self.levels['PDL']['value'] = float(prev_data['low'].min())
            self.levels['PDC']['value'] = float(prev_data.iloc[-1]['close'])

            self.calculate_pivot_point(
                float(prev_data['high'].max()),
                float(prev_data['low'].min()),
                float(prev_data.iloc[-1]['close'])
            )

        if len(prev_dates) >= 2:
            p2_data = self.historical_data[self.historical_data['date'] == prev_dates.iloc[1]]
            self.levels['P2DH']['value'] = float(p2_data['high'].max())
            self.levels['P2DL']['value'] = float(p2_data['low'].min())

        if len(prev_dates) >= 3:
            p3_data = self.historical_data[self.historical_data['date'] == prev_dates.iloc[2]]
            self.levels['P3DH']['value'] = float(p3_data['high'].max())
            self.levels['P3DL']['value'] = float(p3_data['low'].min())

        if not today_data.empty:
            today_data['datetime'] = pd.to_datetime(today_data['timestamp'])
            latest_time = today_data['datetime'].iloc[-1]
            or_end_time = latest_time.replace(hour=10, minute=15)

            self.levels['ORBH']['value'] = 0
            self.levels['ORBL']['value'] = 0

            if latest_time >= or_end_time:
                opening_data = today_data[
                    ((pd.to_datetime(today_data['datetime']).dt.hour == 9) & (pd.to_datetime(today_data['datetime']).dt.minute >= 30)) |
                    ((pd.to_datetime(today_data['datetime']).dt.hour == 10) & (pd.to_datetime(today_data['datetime']).dt.minute <= 15))
                ]

                if not opening_data.empty:
                    self.levels['ORBH']['value'] = float(opening_data['high'].max())
                    self.levels['ORBL']['value'] = float(opening_data['low'].min())
                    if self.debug:
                        print(f"OR Levels set - ORBH: {self.levels['ORBH']['value']:.2f}, ORBL: {self.levels['ORBL']['value']:.2f}")

        if not today_data.empty:
            for span in [8, 17, 24]:
                ema_series = self.calculate_ema(today_data, span)
                if not ema_series.empty:
                    self.levels[f'EMA{span}']['value'] = float(ema_series.iloc[-1])
                    today_data[f'ema{span}'] = ema_series  # Store EMA in DataFrame

            hott, lott = self.calculate_ott(today_data)
            if len(hott) > 0 and len(lott) > 0:
                self.levels['HOTT']['value'] = float(hott.iloc[-1])
                self.levels['LOTT']['value'] = float(lott.iloc[-1])

        self.update_weekly_levels()
        self.update_monthly_levels(reference_date)

    def display_static_levels(self, current_price):
        """Display static price levels"""
        print(f"\n{Fore.CYAN}Static Price Levels (High to Low):{Style.RESET_ALL}")

        static_levels = [
            (k, v) for k, v in self.levels.items()
            if k in ['PPWH', 'PMH', 'PWH', 'PPWL', 'MH', 'ORBH', 'PDH', 'PDC', 'ORBL', 'PDL', 'P2DH', 'P2DL', 'P3DH', 'P3DL', 'PWL', 'PML', 'ML']
        ]

        static_levels.append(('CURRENT', {'desc': 'Current Price', 'value': current_price}))

        static_levels.sort(key=lambda x: x[1]['value'], reverse=True)

        rows = []
        for code, info in static_levels:
            if info['value'] > 0:
                distance = current_price - info['value']
                if code == 'CURRENT':
                    rows.append([
                        f"{Fore.YELLOW}{code}{Style.RESET_ALL}",
                        f"{Fore.YELLOW}{info['desc']}{Style.RESET_ALL}",
                        f"{Fore.YELLOW}${info['value']:.2f}{Style.RESET_ALL}",
                        f"{Fore.YELLOW}0.00{Style.RESET_ALL}"
                    ])
                else:
                    color = Fore.GREEN if distance > 0 else Fore.RED
                    rows.append([
                        code,
                        info['desc'],
                        f"${info['value']:.2f}",
                        f"{color}{distance:+.2f}{Style.RESET_ALL}"
                    ])

        print(tabulate(
            rows,
            headers=['Level', 'Description', 'Price', 'Distance'],
            tablefmt='grid'
        ))

    def display_technical_levels(self, current_price):
        """Display technical levels including EMAs and OTT bands"""
        print(f"\n{Fore.YELLOW}Dynamic Levels:{Style.RESET_ALL}")

        dynamic_data = []

        for span in [8, 17, 24]:
            ema_value = self.levels[f'EMA{span}']['value']
            if ema_value > 0:
                color = Fore.GREEN if current_price > ema_value else Fore.RED
                distance = current_price - ema_value
                dynamic_data.append([
                    f"EMA{span}",
                    f"{color}${ema_value:.2f}{Style.RESET_ALL}",
                    f"{abs(distance):.2f}",
                    "Above" if current_price > ema_value else "Below"
                ])

        if self.debug:
            print(f"Debug - Checking OTT values in levels:")
            print(f"HOTT: {self.levels.get('HOTT', {}).get('value', 'Not found')}")
            print(f"LOTT: {self.levels.get('LOTT', {}).get('value', 'Not found')}")

        for ott_type in ['HOTT', 'LOTT']:
            if ott_type in self.levels and self.levels[ott_type]['value'] > 0:
                ott_value = self.levels[ott_type]['value']
                color = Fore.GREEN if current_price > ott_value else Fore.RED
                distance = current_price - ott_value
                dynamic_data.append([
                    f"{ott_type} (2,1.5,22)",
                    f"{color}${ott_value:.2f}{Style.RESET_ALL}",
                    f"{abs(distance):.2f}",
                    "Above" if current_price > ott_value else "Below"
                ])

        print(tabulate(
            dynamic_data,
            headers=["Indicator", "Value", "Distance", "Position"],
            tablefmt="grid"
        ))

        if self.debug:
            print(f"\nOTT Parameters: Period=2, Percent=1.5, HLLength=22")

    def get_ema_pattern(self, timestamp=None):
        """Determine EMA pattern (ALL STAR or mixed)"""
        if timestamp is None:
            ema8 = self.levels['EMA8']['value']
            ema17 = self.levels['EMA17']['value']
            ema24 = self.levels['EMA24']['value']
        else:
            ema8 = self.get_ema_for_candle(timestamp, 8) or 0
            ema17 = self.get_ema_for_candle(timestamp, 17) or 0
            ema24 = self.get_ema_for_candle(timestamp, 24) or 0
        strength = max(ema8, ema17, ema24) - min(ema8, ema17, ema24)
        if strength < 1:
            return f"{Fore.WHITE}NEUTRAL{Style.RESET_ALL}"
        else:
            if ema8 > ema17 > ema24:
                return f"{Fore.GREEN}ASA Bullish{Style.RESET_ALL}"
            elif ema8 < ema17 < ema24:
                return f"{Fore.RED}ASA Bearish{Style.RESET_ALL}"
            else:
                return f"{Fore.YELLOW}MIX{Style.RESET_ALL}"

    def get_ema_for_candle(self, timestamp, span):
        """Get EMA value for a specific candle"""
        if timestamp in self.ema_history[span]:
            return self.ema_history[span][timestamp]
        return None

    def get_ema_strength(self, timestamp=None):
        """Calculate EMA strength as max-min difference between EMAs"""
        if timestamp is None:
            ema8 = self.levels['EMA8']['value']
            ema17 = self.levels['EMA17']['value']
            ema24 = self.levels['EMA24']['value']
        else:
            ema8 = self.get_ema_for_candle(timestamp, 8) or 0
            ema17 = self.get_ema_for_candle(timestamp, 17) or 0
            ema24 = self.get_ema_for_candle(timestamp, 24) or 0

        if ema8 == 0 or ema17 == 0 or ema24 == 0:
            return "0"

        strength = max(ema8, ema17, ema24) - min(ema8, ema17, ema24)

        if strength <= 1:
            color = Fore.WHITE
        elif strength > 1 and strength <= 4:
            color = Fore.MAGENTA
        elif strength > 4 and strength <= 8:
            color = Fore.CYAN
        else:
            color = Fore.YELLOW

        return f"{color}{strength:.2f}{Style.RESET_ALL}"

    def detect_trend_channel(self, data: pd.DataFrame, lookback: int = 20) -> str:
        """Detect trend channel direction with improved accuracy"""
        if len(data) < lookback:
            return ""

        recent_data = data.tail(lookback)

        x = np.arange(len(recent_data))
        y = recent_data['close'].values
        slope, _ = np.polyfit(x, y, 1)

        price_change_pct = (recent_data['close'].iloc[-1] - recent_data['close'].iloc[0]) / recent_data['close'].iloc[0] * 100

        if slope > 0 and price_change_pct > 0.2:
            return f"{Fore.GREEN}UC{Style.RESET_ALL}"
        elif slope < 0 and price_change_pct < -0.2:
            return f"{Fore.RED}DC{Style.RESET_ALL}"
        return ""

    def detect_breakout(self, data: pd.DataFrame, std_multiplier: float) -> str:
        """Detect price breakouts from recent range"""
        lookback = self.pattern_params['breakout_lookback']
        if len(data) < lookback:
            return ""

        recent = data.iloc[-lookback:-1]
        current = data.iloc[-1]

        mean = recent['close'].mean()
        std = recent['close'].std()
        upper = mean + (std * std_multiplier)
        lower = mean - (std * std_multiplier)

        if current['close'] > upper:
            return f"{Fore.GREEN}UB{Style.RESET_ALL}"
        elif current['close'] < lower:
            return f"{Fore.RED}DB{Style.RESET_ALL}"
        return ""

    def detect_ascending_triangle(self, data: pd.DataFrame, tolerance: float) -> str:
        """Detect ascending triangle pattern"""
        lookback = self.pattern_params['triangle_lookback']
        if len(data) < lookback:
            return ""

        recent = data.iloc[-lookback:]
        highs = recent['high'].values
        lows = recent['low'].values

        max_idx = argrelextrema(highs, np.greater_equal, order=2)[0]
        if len(max_idx) < 2:
            return ""

        last_max_prices = highs[max_idx][-2:]
        if abs(last_max_prices[0] - last_max_prices[1]) <= tolerance:
            xvals = np.arange(len(lows))
            slope_low, _, _, _, _ = linregress(xvals, lows)

            if slope_low > 0:
                return f"{Fore.GREEN}AT{Style.RESET_ALL}"

        return ""

    def detect_head_shoulders(self, data: pd.DataFrame, price_tolerance: float) -> str:
        """Detect head and shoulders patterns"""
        MIN_CANDLES_FOR_HS = 30

        if len(data) < MIN_CANDLES_FOR_HS:
            return ""

        try:
            highs_smooth = savgol_filter(data['high'].values, window_length=5, polyorder=2)
            lows_smooth = savgol_filter(data['low'].values, window_length=5, polyorder=2)
        except Exception:
            highs_smooth = data['high'].values
            lows_smooth = data['low'].values

        max_idx = argrelextrema(highs_smooth, np.greater_equal, order=2)[0]
        min_idx = argrelextrema(lows_smooth, np.less_equal, order=2)[0]

        if len(max_idx) >= 3:
            last_three_peaks = highs_smooth[max_idx[-3:]]

            if (last_three_peaks[1] > last_three_peaks[0] and
                last_three_peaks[1] > last_three_peaks[2] and
                abs(last_three_peaks[0] - last_three_peaks[2]) < price_tolerance):
                return f"{Fore.RED}HS{Style.RESET_ALL}"

        if len(min_idx) >= 3:
            last_three_troughs = lows_smooth[min_idx[-3:]]

            if (last_three_troughs[1] < last_three_troughs[0] and
                last_three_troughs[1] < last_three_troughs[2] and
                abs(last_three_troughs[0] - last_three_troughs[2]) < price_tolerance):
                return f"{Fore.GREEN}IHS{Style.RESET_ALL}"

        return ""

    def detect_all_patterns(self, data: pd.DataFrame) -> dict:
        """Detect all patterns using pattern_detector_spx logic"""
        MIN_CANDLES_FOR_PATTERN = 15

        if self.debug:
            print("\nPattern Detection:")
            print(f"- Candles available: {len(data)}")
            print(f"- Minimum needed: {MIN_CANDLES_FOR_PATTERN}")

        if len(data) < MIN_CANDLES_FOR_PATTERN:
            if self.debug:
                print("- Skipping pattern detection (not enough data)")
            return {}

        patterns = {}

        if self.debug:
            print("\nChecking patterns:")

        if self.debug:
            print("1. Trend Channel...")
        patterns['trend'] = self.detect_trend_channel(data, lookback=30)

        if self.debug:
            print("2. Head & Shoulders...")
        if len(data) >= 30:
            patterns['reversal'] = self.detect_head_shoulders(data, price_tolerance=10.0)

        if self.debug:
            print("3. Ascending Triangle...")
        if len(data) >= 30:
            patterns['continuation'] = self.detect_ascending_triangle(data, tolerance=10.0)

        if self.debug:
            print("4. Breakout...")
        if len(data) >= 30:
            patterns['breakout'] = self.detect_breakout(data, std_multiplier=1.8)

        if self.debug:
            print(f"\nPatterns found: {patterns}")

        return {k: v for k, v in patterns.items() if v}

    def format_patterns(self, patterns):
        """Format patterns for display"""
        if not patterns:
            return '', ''

        primary = ''
        supporting = []

        if patterns.get('breakout'):
            primary = patterns['breakout']
            if patterns.get('reversal'): supporting.append(patterns['reversal'])
            if patterns.get('continuation'): supporting.append(patterns['continuation'])
            if patterns.get('trend'): supporting.append(patterns['trend'])
        elif patterns.get('reversal'):
            primary = patterns['reversal']
            if patterns.get('continuation'): supporting.append(patterns['continuation'])
            if patterns.get('trend'): supporting.append(patterns['trend'])
        elif patterns.get('continuation'):
            primary = patterns['continuation']
            if patterns.get('trend'): supporting.append(patterns['trend'])
        elif patterns.get('trend'):
            primary = patterns['trend']

        return primary, ' || '.join(supporting) if supporting else ''

    def add_htf_impulse_trend(self, data: pd.DataFrame) -> pd.Series:
        """Adds HTF_Impulse_Trend column using last 3 completed 10-min candle closes."""
        trend_list = []

        for i in range(len(data)):
            if i < 30:
                trend_list.append("")
                continue

            c1 = data.iloc[i-30:i-20]['close'].iloc[-1]
            c2 = data.iloc[i-20:i-10]['close'].iloc[-1]
            c3 = data.iloc[i-10:i]['close'].iloc[-1]

            if c1 < c2 < c3:
                trend_list.append("Up Impulse")
            elif c1 > c2 > c3:
                trend_list.append("Down Impulse")
            else:
                trend_list.append("Choppy")

        return pd.Series(trend_list, name="HTF_Impulse_Trend")

    def add_htf_sweep_flag(self, data: pd.DataFrame) -> pd.Series:
        """Adds HTF_Sweep_Flag column to detect sweeps (false breakouts) of 30-min high/low."""
        sweep_list = []

        for i in range(len(data)):
            if i < 30:
                sweep_list.append("")
                continue

            window = data.iloc[i-30:i]
            high_30 = window['high'].max()
            low_30 = window['low'].min()

            high_now = data.iloc[i]['high']
            low_now = data.iloc[i]['low']
            close_now = data.iloc[i]['close']

            if high_now > high_30 and close_now < high_30:
                sweep_list.append("SweepH")
            elif low_now < low_30 and close_now > low_30:
                sweep_list.append("SweepL")
            else:
                sweep_list.append("")

        return pd.Series(sweep_list, name="HTF_Sweep_Flag")

    def add_htf_zone_tag(self, data: pd.DataFrame) -> pd.Series:
        """Adds HTF_Zone_Tag column showing if current close is near 30-min high/low."""
        zone_list = []

        for i in range(len(data)):
            if i < 30:
                zone_list.append("")
                continue

            window = data.iloc[i-30:i]
            high_30 = window['high'].max()
            low_30 = window['low'].min()
            close_now = data.iloc[i]['close']

            near_high = abs(close_now - high_30) <= 2
            near_low = abs(close_now - low_30) <= 2

            if near_high and near_low:
                zone_list.append("Confluence")
            elif near_high:
                zone_list.append("Near 30H")
            elif near_low:
                zone_list.append("Near 30L")
            else:
                zone_list.append("")

        return pd.Series(zone_list, name="HTF_Zone_Tag")

    def run(self, mode='live', target_date=None, step_through=False):
        """Run the analyzer in specified mode"""
        if self.debug:
            print("\n=== Starting run() function ===")
            print(f"Mode: {mode}, Target date: {target_date}")

        self.load_historical_data(target_date)

        if self.debug:
            print("\n=== Processing historical data ===")
            print(f"Data shape: {self.historical_data.shape}")
            print("Available dates:", sorted(self.historical_data['date'].unique()))

        if mode == 'live':
            try:
                today = datetime.now(self.est)
                today_str = today.strftime('%Y%m%d')

                self.update_weekly_levels()
                self.update_monthly_levels()

                prev_date = self.historical_data[self.historical_data['date'] < today_str]['date'].max()
                if pd.notna(prev_date):
                    prev_data = self.historical_data[self.historical_data['date'] == prev_date]
                    self.levels['PDH']['value'] = float(prev_data['high'].max())
                    self.levels['PDL']['value'] = float(prev_data['low'].min())
                    self.levels['PDC']['value'] = float(prev_data.iloc[-1]['close'])

                last_file_mtime = 0
                last_processed_time = None
                today_data = pd.DataFrame()

                while True:
                    try:
                        year = today_str[:4]
                        month_num = today_str[4:6]
                        month_map = {
                            '01': 'JAN', '02': 'FEB', '03': 'MAR', '04': 'APR',
                            '05': 'MAY', '06': 'JUN', '07': 'JUL', '08': 'AUG',
                            '09': 'SEP', '10': 'OCT', '11': 'NOV', '12': 'DEC'
                        }
                        month = month_map[month_num]
                        file_path = self.data_dir / year / month / f"spx_{today_str[6:8]}{month_num}{today_str[2:4]}.csv"

                        current_mtime = file_path.stat().st_mtime if file_path.exists() else 0

                        if current_mtime > last_file_mtime:
                            if today_data.empty:
                                today_data = pd.read_csv(file_path)
                                today_data['date'] = today_str
                                # Convert timestamp to datetime
                                today_data['timestamp'] = pd.to_datetime(today_data['timestamp'])
                                # Resample to 5-minute candles
                                today_data = today_data.resample('5T', on='timestamp').agg({
                                    'open': 'first',
                                    'high': 'max',
                                    'low': 'min',
                                    'close': 'last',
                                    'date': 'first'
                                }).reset_index()
                            else:
                                latest_data = pd.read_csv(file_path).iloc[-1:]
                                latest_data['date'] = today_str
                                latest_data['timestamp'] = pd.to_datetime(latest_data['timestamp'])

                                if latest_data['timestamp'].iloc[0] != last_processed_time:
                                    # Add new data and resample
                                    today_data = pd.concat([today_data, latest_data], ignore_index=True)
                                    today_data = today_data.resample('5T', on='timestamp').agg({
                                        'open': 'first',
                                        'high': 'max',
                                        'low': 'min',
                                        'close': 'last',
                                        'date': 'first'
                                    }).reset_index()
                                else:
                                    today_data.iloc[-1] = latest_data.iloc[0]

                            last_file_mtime = current_mtime
                            last_processed_time = today_data['timestamp'].iloc[-1]

                            if not today_data.empty:
                                latest_data = today_data.iloc[-1]
                                current_price = latest_data['close']

                                current_time = pd.to_datetime(latest_data['timestamp'])
                                or_end_time = current_time.replace(hour=10, minute=15)

                                if current_time >= or_end_time and self.levels['ORBH']['value'] == 0:
                                    opening_data = today_data[
                                        ((pd.to_datetime(today_data['timestamp']).dt.hour == 9) & (pd.to_datetime(today_data['timestamp']).dt.minute >= 30)) |
                                        ((pd.to_datetime(today_data['timestamp']).dt.hour == 10) & (pd.to_datetime(today_data['timestamp']).dt.minute <= 15))
                                    ]
                                    if not opening_data.empty:
                                        self.levels['ORBH']['value'] = float(opening_data['high'].max())
                                        self.levels['ORBL']['value'] = float(opening_data['low'].min())

                                for span in [8, 17, 24]:
                                    ema_series = self.calculate_ema(today_data, span)
                                    if not ema_series.empty:
                                        self.levels[f'EMA{span}']['value'] = float(ema_series.iloc[-1])
                                        today_data[f'ema{span}'] = ema_series  # Store EMA in DataFrame

                                hott, lott = self.calculate_ott(today_data)
                                if len(hott) > 0 and len(lott) > 0:
                                    self.levels['HOTT']['value'] = float(hott.iloc[-1])
                                    self.levels['LOTT']['value'] = float(lott.iloc[-1])

                                print('\033[2J\033[H')
                                print(f"\n{Fore.CYAN}SPX Analysis - {latest_data['date']} {latest_data['timestamp']}{Style.RESET_ALL}")
                                print(f"Current Price: ${current_price:.2f}")

                                self.display_static_levels(current_price)
                                self.display_technical_levels(current_price)
                                self.display_candle_analysis(today_data)

                                last10 = today_data.tail(10)
                                ema8 = last10['ema8'].values
                                x = np.arange(len(ema8))
                                slope, _ = np.polyfit(x, ema8, 1)
                                print(f"Recent EMA8 slope (trend rate) over last 10 candles: {slope:.4f} points/candle")

                                print(f"\nLast update: {datetime.now(self.est).strftime('%H:%M:%S')} ET")
                                print("Monitoring for updates... (Press Ctrl+C to stop)")

                        time.sleep(0.1)

                    except FileNotFoundError:
                        print(f"{Fore.RED}Waiting for today's data file...{Style.RESET_ALL}")
                        time.sleep(5)
                        continue
                    except Exception as e:
                        print(f"{Fore.RED}Error in live update: {str(e)}{Style.RESET_ALL}")
                        time.sleep(1)
                        continue

            except KeyboardInterrupt:
                print("\nScanner stopped by user")

        elif mode == 'historical':
            if target_date:
                self.update_all_levels(target_date)
                target_data = self.historical_data[self.historical_data['date'] == target_date].reset_index(drop=True)
                if not target_data.empty and hasattr(target_data['timestamp'].iloc[0], 'tzinfo'):
                    target_data['timestamp'] = target_data['timestamp'].apply(lambda x: x.replace(tzinfo=None) if hasattr(x, 'tzinfo') and x.tzinfo else x)
                self.historical_data['timestamp'] = self.historical_data['timestamp'].astype(str)
                target_data['timestamp'] = target_data['timestamp'].astype(str)
                for i in range(len(target_data)):
                    ts = target_data.iloc[i]['timestamp']
                    for span in [8, 17, 24]:
                        self.ema_history[span][ts] = self.calculate_ema(target_data.iloc[:i+1], span).iloc[-1]
                if not target_data.empty:
                    if step_through:
                        for i in range(len(target_data)):
                            data_slice = target_data.iloc[:i+1]
                            candle = data_slice.iloc[-1]
                    else:
                        print(f"\n{Fore.CYAN}SPX Historical Analysis - {target_date}{Style.RESET_ALL}")
                        print(f"Total candles: {len(target_data)}")
                        current_price = target_data.iloc[-1]['close']
                        self.display_static_levels(current_price)
                        self.display_technical_levels(current_price)
                        self.display_candle_analysis(target_data)
                        input("\nPress Enter to exit...")
                else:
                    print(f"{Fore.RED}No data found for date {target_date}{Style.RESET_ALL}")
            time.sleep(10)

        elif mode == 'simulation':
            if target_date:
                target_data = self.historical_data[self.historical_data['date'] == target_date]
                for i in range(len(target_data)):
                    ts = target_data.iloc[i]['timestamp']
                    for span in [8, 17, 24]:
                        self.ema_history[span][ts] = self.calculate_ema(target_data.iloc[:i+1], span).iloc[-1]
                if not target_data.empty:
                    candle_index = 0
                    while candle_index < len(target_data):
                        print('\033[2J\033[H')
                        self.update_all_levels(target_date, candle_index)
                        current_data = target_data.iloc[:candle_index + 1]
                        current_price = current_data.iloc[-1]['close']

                        print(f"\n{Fore.CYAN}SPX Simulation Analysis - {target_date}{Style.RESET_ALL}")
                        print(f"Current Price: ${current_price:.2f}")
                        print(f"Candle: {candle_index + 1}/{len(target_data)}")

                        self.display_static_levels(current_price)
                        self.display_technical_levels(current_price)

                        self.display_candle_analysis(current_data)

                        if candle_index < len(target_data) - 1:
                            input("\nPress Enter for next candle...")
                            candle_index += 1
                        else:
                            print("\nReached end of day")
                            input("Press Enter to exit...")
                            break
                else:
                    print(f"{Fore.RED}No data found for date {target_date}{Style.RESET_ALL}")

        elif mode == 'quick_view':
            if target_date:
                self.update_all_levels(target_date)
                target_data = self.historical_data[self.historical_data['date'] == target_date].reset_index(drop=True)

                for i in range(len(target_data)):
                    ts = target_data.iloc[i]['timestamp']
                    for span in [8, 17, 24]:
                        self.ema_history[span][ts] = self.calculate_ema(target_data.iloc[:i+1], span).iloc[-1]

                if not target_data.empty:
                    print(f"\n{Fore.CYAN}SPX Quick View Analysis - {target_date}{Style.RESET_ALL}")
                    current_price = target_data.iloc[-1]['close']

                    self.display_static_levels(current_price)
                    self.display_technical_levels(current_price)

                    print(f"\n{Fore.CYAN}Last 10 Candles:{Style.RESET_ALL}")
                    self.display_candle_analysis(target_data.tail(10))
                else:
                    print(f"{Fore.RED}No data found for date {target_date}{Style.RESET_ALL}")
                time.sleep(10)

    def calculate_ema(self, series, span):
        """
        Calculate the Exponential Moving Average (EMA) for a pandas Series or DataFrame.
        If a DataFrame is passed, use the 'close' column.
        :param series: pandas Series of prices or DataFrame with 'close' column
        :param span: int, the span for EMA
        :return: pandas Series of EMA values
        """
        if isinstance(series, pd.DataFrame):
            if 'close' in series.columns:
                price_series = pd.to_numeric(series['close'], errors='coerce')
            else:
                raise ValueError("DataFrame must contain a 'close' column for EMA calculation.")
        else:
            price_series = pd.to_numeric(series, errors='coerce')
        return price_series.ewm(span=span, adjust=False).mean()

    def get_candle_analysis_table(self, data):
        """Return candle-by-candle analysis as a DataFrame for web display."""
        data = data.copy()
        if 'shendi' not in data.columns:
            data['shendi'] = data.apply(lambda row: row['high'] - max(row['open'], row['close']), axis=1)
        if 'shepti' not in data.columns:
            data['shepti'] = data.apply(lambda row: min(row['open'], row['close']) - row['low'], axis=1)
        if 'ema8' not in data.columns:
            data['ema8'] = data['close'].ewm(span=8, adjust=False).mean()
        if 'ema21' not in data.columns:
            data['ema21'] = data['close'].ewm(span=21, adjust=False).mean()
        if 'ema55' not in data.columns:
            data['ema55'] = data['close'].ewm(span=55, adjust=False).mean()
        if 'price_vs_ema' not in data.columns:
            def price_vs_ema_fn(row):
                if row['close'] > row['ema8'] and row['close'] > row['ema21'] and row['close'] > row['ema55']:
                    return "Above All"
                elif row['close'] < row['ema8'] and row['close'] < row['ema21'] and row['close'] < row['ema55']:
                    return "Below All"
                else:
                    return "Mixed"
            data['price_vs_ema'] = data.apply(price_vs_ema_fn, axis=1)
        # Add more columns as needed for your analysis
        # You can add the same columns as in display_candle_analysis
        return data

    def get_candle_analysis_styled_table(self, data):
        """Return a styled DataFrame for Streamlit with color coding and 2 decimal formatting, and without 'date' column."""
        df = self.get_candle_analysis_table(data)
        # Remove 'date' column if present
        if 'date' in df.columns:
            df = df.drop(columns=['date'])
        # Format all float columns to 2 decimals
        float_cols = df.select_dtypes(include=['float', 'float64', 'float32']).columns
        df[float_cols] = df[float_cols].applymap(lambda x: f"{x:.2f}" if pd.notnull(x) else x)
        def highlight_body(val):
            try:
                v = float(val)
                if v > 0:
                    return 'background-color: #e6ffe6; color: green; font-weight: bold;'
                elif v < 0:
                    return 'background-color: #ffe6e6; color: red; font-weight: bold;'
            except:
                pass
            return ''
        def highlight_trend(val):
            if 'Above' in str(val):
                return 'background-color: #e6ffe6; color: green; font-weight: bold;'
            elif 'Below' in str(val):
                return 'background-color: #ffe6e6; color: red; font-weight: bold;'
            elif 'Mixed' in str(val):
                return 'background-color: #ffffe6; color: orange; font-weight: bold;'
            return ''
        def highlight_high(val, all_highs):
            try:
                v = float(val)
                if v == max([float(x) for x in all_highs]):
                    return 'background-color: #e6ffe6; color: green; font-weight: bold;'
            except:
                pass
            return ''
        def highlight_low(val, all_lows):
            try:
                v = float(val)
                if v == min([float(x) for x in all_lows]):
                    return 'background-color: #ffe6e6; color: red; font-weight: bold;'
            except:
                pass
            return ''
        styled = df.style
        if 'Body' in df.columns:
            styled = styled.applymap(highlight_body, subset=['Body'])
        if 'Trend' in df.columns:
            styled = styled.applymap(highlight_trend, subset=['Trend'])
        if 'High' in df.columns:
            styled = styled.apply(lambda col: [highlight_high(v, df['High']) for v in col], subset=['High'])
        if 'Low' in df.columns:
            styled = styled.apply(lambda col: [highlight_low(v, df['Low']) for v in col], subset=['Low'])
        return styled

def main():
    print("\nSelect Mode:")
    print("1. Live Mode (Real-time updates)")
    print("2. Historical Mode (Complete analysis)")
    print("3. Simulation Mode (Step through candles)")
    print("4. Today's Analysis (One-time analysis)")
    print("5. Candle-by-Candle Analysis (Detailed debugging)")
    print("6. Quick View (Last 10 candles)")
    
    choice = input("\nEnter your choice (1-6): ")
    debug_mode = (choice == "5")
    scanner = SPX5MinCandleAnalyzer(debug=debug_mode)
    
    try:
        if choice == "1":
            scanner.run(mode='live')
        elif choice == "4":
            today_date = datetime.now(scanner.est).strftime('%Y%m%d')
            scanner.run(mode='historical', target_date=today_date)
            sys.exit()
        elif choice == "6":
            date_str = input("\nEnter date (YYYYMMDD format, e.g., 20250130): ")
            if len(date_str) == 8 and date_str.isdigit():
                try:
                    datetime.strptime(date_str, '%Y%m%d')
                    scanner.run(mode='quick_view', target_date=date_str)
                    sys.exit()
                except ValueError:
                    print(f"{Fore.RED}Invalid date format. Please use YYYYMMDD format.{Style.RESET_ALL}")
        elif choice in ["2", "3", "5"]:
            date_str = input("\nEnter date (YYYYMMDD format, e.g., 20250130): ")
            if len(date_str) == 8 and date_str.isdigit():
                try:
                    datetime.strptime(date_str, '%Y%m%d')
                    if choice == "2":
                        scanner.run(mode='historical', target_date=date_str, step_through=False)
                    elif choice == "5":
                        scanner.run(mode='historical', target_date=date_str, step_through=True)
                    else:
                        scanner.run(mode='simulation', target_date=date_str)
                    
                    if choice == "2":
                        sys.exit()
                except ValueError:
                    print(f"{Fore.RED}Invalid date format. Please use YYYYMMDD format.{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}Invalid date format. Please use YYYYMMDD format.{Style.RESET_ALL}")
        else:
            print("Invalid choice. Please enter 1, 2, 3, 4, 5, or 6.")
    except KeyboardInterrupt:
        print("\nProgram terminated by user")
        sys.exit()

if __name__ == "__main__":
    main()