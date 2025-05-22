# SPX 5-Minute Analyzer Dashboard

A Streamlit dashboard for analyzing SPX 5-minute candle data, providing real-time insights and historical analysis.

## Features

- Real-time SPX 5-minute candle analysis
- Historical data analysis
- Technical indicators including EMAs and OTT
- Static and dynamic price levels
- Higher timeframe analysis
- Color-coded price movements

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/spx-analyzer.git
cd spx-analyzer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the dashboard:
```bash
streamlit run spx_5min_dashboard.py
```

## Usage

1. Enter the date in YYYYMMDD format
2. Choose the analysis mode:
   - Historical: Full day analysis
   - Quick View: Last 10 candles
   - Simulation: Step through candles

## Requirements

- Python 3.8+
- Streamlit
- Pandas
- NumPy
- Tabulate 