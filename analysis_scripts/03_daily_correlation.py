import pandas as pd
import numpy as np
from scipy import stats

merged = pd.read_pickle('E:\\Analysis_Support\\merged.pkl')
merged = merged.dropna(subset=['sentiment']).copy()

daily = merged.groupby('date').agg(
    n_trades=('closed_pnl', 'size'),
    total_pnl=('closed_pnl', 'sum'),
    total_volume=('size_usd', 'sum'),
    win_rate=('closed_pnl', lambda x: (x[x != 0] > 0).mean() * 100 if (x != 0).any() else np.nan),
    avg_trade_size=('size_usd', 'mean'),
    fg_value=('fg_value', 'first'),
    n_accounts=('account', 'nunique'),
).reset_index()

daily['pnl_per_volume_bps'] = daily['total_pnl'] / daily['total_volume'] * 10000

# Correlations between continuous FG value and daily metrics
print("="*90)
print("CORRELATION: daily Fear&Greed index value vs daily trading metrics (Spearman)")
print("="*90)
for col in ['n_trades', 'total_pnl', 'total_volume', 'win_rate', 'avg_trade_size', 'pnl_per_volume_bps']:
    valid = daily[['fg_value', col]].dropna()
    rho, p = stats.spearmanr(valid['fg_value'], valid[col])
    print(f"  fg_value vs {col:25s}: rho={rho:+.3f}  p={p:.4g}  n={len(valid)}")

daily.to_csv('E:\\Analysis_Support\\data_exports\\daily_metrics.csv', index=False)

# Lagged effect: does yesterday's sentiment predict today's PnL / volume?
daily_sorted = daily.sort_values('date').reset_index(drop=True)
daily_sorted['fg_value_lag1'] = daily_sorted['fg_value'].shift(1)
valid = daily_sorted[['fg_value_lag1', 'total_pnl', 'total_volume']].dropna()
rho_pnl, p_pnl = stats.spearmanr(valid['fg_value_lag1'], valid['total_pnl'])
rho_vol, p_vol = stats.spearmanr(valid['fg_value_lag1'], valid['total_volume'])
print("\n" + "="*90)
print("LAGGED EFFECT: yesterday's FG value vs today's outcomes")
print("="*90)
print(f"  lag1 fg_value vs today total_pnl:    rho={rho_pnl:+.3f}  p={p_pnl:.4g}")
print(f"  lag1 fg_value vs today total_volume: rho={rho_vol:+.3f}  p={p_vol:.4g}")

print("\nSaved daily_metrics.csv")
