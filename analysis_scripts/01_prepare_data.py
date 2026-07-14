"""
Step 1: Load, clean and merge the two datasets.
- Bitcoin Fear & Greed sentiment index (daily)
- Hyperliquid historical trader execution data (per-trade)
"""
import pandas as pd
import numpy as np

pd.set_option('display.width', 140)

# ---------- Load sentiment data ----------
fg = pd.read_csv('E:\\Analysis_Support\\fear_greed_index.csv')
fg['date'] = pd.to_datetime(fg['date'])
fg = fg.rename(columns={'value': 'fg_value', 'classification': 'sentiment'})
fg = fg[['date', 'fg_value', 'sentiment']].sort_values('date').reset_index(drop=True)

# Collapse 5-class sentiment into a simpler 3-class version for some views
sent_map_3 = {
    'Extreme Fear': 'Fear',
    'Fear': 'Fear',
    'Neutral': 'Neutral',
    'Greed': 'Greed',
    'Extreme Greed': 'Greed',
}
fg['sentiment_3'] = fg['sentiment'].map(sent_map_3)

# ---------- Load trader data ----------
tr = pd.read_csv('E:\\Analysis_Support\\historical_data.csv', encoding='utf-8')

tr['datetime'] = pd.to_datetime(tr['Timestamp IST'], format='mixed', dayfirst=True)
# Sanity check: flag if any timestamps failed to parse
n_bad = tr['datetime'].isna().sum()
if n_bad:
    print(f"WARNING: {n_bad} timestamps failed to parse and are NaT")
tr['date'] = tr['datetime'].dt.floor('D')

# Standardize column names we will use
tr = tr.rename(columns={
    'Account': 'account',
    'Coin': 'coin',
    'Execution Price': 'exec_price',
    'Size Tokens': 'size_tokens',
    'Size USD': 'size_usd',
    'Side': 'side',
    'Start Position': 'start_position',
    'Direction': 'direction',
    'Closed PnL': 'closed_pnl',
    'Fee': 'fee',
    'Crossed': 'crossed',
    'Trade ID': 'trade_id',
})

keep_cols = ['account', 'coin', 'exec_price', 'size_tokens', 'size_usd', 'side',
             'direction', 'start_position', 'closed_pnl', 'fee', 'crossed',
             'trade_id', 'datetime', 'date']
tr = tr[keep_cols]

# Net PnL after fees for closing trades
tr['net_pnl'] = tr['closed_pnl'] - tr['fee']
tr['is_close'] = tr['direction'].astype(str).str.contains('Close', case=False, na=False) | \
                  tr['direction'].isin(['Buy', 'Sell'])
tr['is_win'] = tr['closed_pnl'] > 0
tr['is_loss'] = tr['closed_pnl'] < 0
tr['has_realized_pnl'] = tr['closed_pnl'] != 0

# ---------- Merge on date ----------
merged = tr.merge(fg[['date', 'fg_value', 'sentiment', 'sentiment_3']], on='date', how='left')

print("Trader rows:", len(tr))
print("Rows matched to a sentiment day:", merged['sentiment'].notna().sum())
print("Rows NOT matched (date outside FG index range):", merged['sentiment'].isna().sum())
print("Trader date range:", tr['date'].min(), "->", tr['date'].max())
print("FG index date range:", fg['date'].min(), "->", fg['date'].max())

merged.to_pickle('E:\\Analysis_Support\\data_exports\\merged.pkl')
fg.to_pickle('E:\\Analysis_Support\\data_exports\\fg.pkl')
print("\nSaved merged.pkl and fg.pkl")
print(merged['sentiment'].value_counts())
