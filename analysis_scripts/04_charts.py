import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.edgecolor': '#444444',
    'axes.labelcolor': '#222222',
    'text.color': '#222222',
    'xtick.color': '#222222',
    'ytick.color': '#222222',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})

C = {
    'Extreme Fear': '#8c2d19',
    'Fear': '#d9642c',
    'Neutral': '#9e9e9e',
    'Greed': '#5a9c5f',
    'Extreme Greed': '#1b6b3a',
}
sent_order5 = ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']

OUT = 'E:\\Analysis_Support\\charts'

merged = pd.read_pickle('E:\\Analysis_Support\\merged.pkl')
merged = merged.dropna(subset=['sentiment']).copy()
merged['sentiment'] = pd.Categorical(merged['sentiment'], categories=sent_order5, ordered=True)

m5 = pd.read_csv('E:\\Analysis_Support\\data_exports\\metrics_by_sentiment5.csv', index_col=0)
m5 = m5.reindex(sent_order5)
daily = pd.read_csv('E:\\Analysis_Support\\data_exports\\daily_metrics.csv', parse_dates=['date'])
bias = pd.read_csv('E:\\Analysis_Support\\data_exports\\long_short_bias.csv', index_col=0).reindex(sent_order5)

def style_ax(ax):
    ax.grid(axis='y', color='#e5e5e5', linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

# ---------------------------------------------------------------
# Chart 1: Win rate by sentiment
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 4.5))
colors = [C[s] for s in sent_order5]
bars = ax.bar(sent_order5, m5['win_rate_pct'], color=colors, zorder=3, width=0.62)
for b, v in zip(bars, m5['win_rate_pct']):
    ax.text(b.get_x()+b.get_width()/2, v+0.8, f"{v:.1f}%", ha='center', fontsize=10, fontweight='bold')
style_ax(ax)
ax.set_ylabel('Win rate (% of closed trades with profit)')
ax.set_title('Trader Win Rate by Market Sentiment', fontsize=13, fontweight='bold', loc='left')
ax.set_ylim(0, 100)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(f'{OUT}/01_win_rate_by_sentiment.png', dpi=150)
plt.close()

# ---------------------------------------------------------------
# Chart 2: Total closed PnL by sentiment
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 4.5))
vals = m5['total_closed_pnl'] / 1e6
bars = ax.bar(sent_order5, vals, color=colors, zorder=3, width=0.62)
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v+0.03, f"${v:.2f}M", ha='center', fontsize=10, fontweight='bold')
style_ax(ax)
ax.set_ylabel('Total closed PnL (USD, millions)')
ax.set_title('Total Trader Profit by Market Sentiment', fontsize=13, fontweight='bold', loc='left')
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(f'{OUT}/02_total_pnl_by_sentiment.png', dpi=150)
plt.close()

# ---------------------------------------------------------------
# Chart 3: Risk-adjusted profitability (PnL per $ volume, bps)
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 4.5))
vals = m5['pnl_per_$_volume_bps']
bars = ax.bar(sent_order5, vals, color=colors, zorder=3, width=0.62)
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v+3, f"{v:.0f} bps", ha='center', fontsize=10, fontweight='bold')
style_ax(ax)
ax.set_ylabel('PnL per $ traded (basis points)')
ax.set_title('Capital Efficiency: Profit per Dollar Traded, by Sentiment', fontsize=13, fontweight='bold', loc='left')
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(f'{OUT}/03_pnl_per_volume_by_sentiment.png', dpi=150)
plt.close()

# ---------------------------------------------------------------
# Chart 4: Long vs Short bias by sentiment (stacked)
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 4.5))
x = np.arange(len(sent_order5))
long_pct = bias['Open Long']
short_pct = bias['Open Short']
ax.bar(x, long_pct, color='#2e7d32', label='Open Long', width=0.55, zorder=3)
ax.bar(x, short_pct, bottom=long_pct, color='#c62828', label='Open Short', width=0.55, zorder=3)
for i, (l, s) in enumerate(zip(long_pct, short_pct)):
    ax.text(i, l/2, f"{l:.0f}%", ha='center', va='center', color='white', fontsize=9, fontweight='bold')
    ax.text(i, l+s/2, f"{s:.0f}%", ha='center', va='center', color='white', fontsize=9, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(sent_order5, rotation=15)
ax.set_ylabel('% of new positions opened')
ax.set_title('Long vs Short Positioning by Market Sentiment', fontsize=13, fontweight='bold', loc='left')
ax.legend(loc='upper right', frameon=False)
style_ax(ax)
plt.tight_layout()
plt.savefig(f'{OUT}/04_long_short_bias.png', dpi=150)
plt.close()

# ---------------------------------------------------------------
# Chart 5: Average trade size by sentiment
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 4.5))
vals = m5['avg_trade_size_usd']
bars = ax.bar(sent_order5, vals, color=colors, zorder=3, width=0.62)
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v+60, f"${v:,.0f}", ha='center', fontsize=10, fontweight='bold')
style_ax(ax)
ax.set_ylabel('Average trade size (USD)')
ax.set_title('Average Trade Size by Market Sentiment', fontsize=13, fontweight='bold', loc='left')
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(f'{OUT}/05_avg_trade_size_by_sentiment.png', dpi=150)
plt.close()

# ---------------------------------------------------------------
# Chart 6: Time series — FG index value with 7d volume overlay
# ---------------------------------------------------------------
daily = daily.sort_values('date')
daily['volume_7d'] = daily['total_volume'].rolling(7, min_periods=1).mean()
daily['fg_7d'] = daily['fg_value'].rolling(7, min_periods=1).mean()

fig, ax1 = plt.subplots(figsize=(9.5, 4.8))
ax1.fill_between(daily['date'], daily['fg_7d'], color='#8c2d19', alpha=0.15)
ax1.plot(daily['date'], daily['fg_7d'], color='#8c2d19', linewidth=1.6, label='Fear & Greed Index (7d avg)')
ax1.set_ylabel('Fear & Greed Index (0-100)', color='#8c2d19')
ax1.tick_params(axis='y', labelcolor='#8c2d19')
ax1.set_ylim(0, 100)

ax2 = ax1.twinx()
ax2.plot(daily['date'], daily['volume_7d']/1e6, color='#1b4f8c', linewidth=1.4, label='Trading volume (7d avg, $M)')
ax2.set_ylabel('Daily trading volume ($M, 7d avg)', color='#1b4f8c')
ax2.tick_params(axis='y', labelcolor='#1b4f8c')

ax1.set_title('Market Sentiment vs Trading Volume Over Time', fontsize=13, fontweight='bold', loc='left')
ax1.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)
fig.tight_layout()
plt.savefig(f'{OUT}/06_sentiment_vs_volume_timeseries.png', dpi=150)
plt.close()

# ---------------------------------------------------------------
# Chart 7: Boxplot of closed PnL by sentiment (trimmed for visibility)
# ---------------------------------------------------------------
realized = merged[merged['has_realized_pnl']].copy()
# trim extreme outliers for visual clarity (winsorize at 1st/99th pct)
lo, hi = realized['closed_pnl'].quantile([0.02, 0.98])
trimmed = realized[(realized['closed_pnl'] >= lo) & (realized['closed_pnl'] <= hi)]

fig, ax = plt.subplots(figsize=(7.5, 4.8))
data = [trimmed.loc[trimmed['sentiment'] == s, 'closed_pnl'].values for s in sent_order5]
bp = ax.boxplot(data, label=sent_order5, patch_artist=True, showfliers=False, widths=0.5)
for patch, s in zip(bp['boxes'], sent_order5):
    patch.set_facecolor(C[s])
    patch.set_alpha(0.7)
for median in bp['medians']:
    median.set_color('black')
ax.axhline(0, color='#888888', linewidth=0.8, linestyle='--', zorder=1)
style_ax(ax)
ax.set_ylabel('Closed PnL per trade (USD, middle 96% shown)')
ax.set_title('Distribution of Trade-Level PnL by Sentiment', fontsize=13, fontweight='bold', loc='left')
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(f'{OUT}/07_pnl_distribution_boxplot.png', dpi=150)
plt.close()

# ---------------------------------------------------------------
# Chart 8: Heatmap — top coins x sentiment mean PnL
# ---------------------------------------------------------------
coin_sent = pd.read_csv('E:\\Analysis_Support\\coin_pnl_by_sentiment.csv')
pivot = coin_sent.pivot(index='coin', columns='sentiment_3', values='mean')
pivot = pivot[['Fear', 'Neutral', 'Greed']]
order = merged['coin'].value_counts().head(8).index.tolist()
pivot = pivot.reindex(order)

fig, ax = plt.subplots(figsize=(6.5, 5))
vmax = np.nanmax(np.abs(pivot.values))
im = ax.imshow(pivot.values, cmap='RdYlGn', vmin=-vmax, vmax=vmax, aspect='auto')
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels(pivot.columns)
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index)
for i in range(pivot.shape[0]):
    for j in range(pivot.shape[1]):
        val = pivot.values[i, j]
        if not np.isnan(val):
            ax.text(j, i, f"${val:,.0f}", ha='center', va='center', fontsize=9,
                    color='black', fontweight='bold')
ax.set_title('Average PnL per Trade by Coin and Sentiment', fontsize=13, fontweight='bold', loc='left')
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Avg PnL per trade (USD)')
plt.tight_layout()
plt.savefig(f'{OUT}/08_coin_sentiment_heatmap.png', dpi=150)
plt.close()

# ---------------------------------------------------------------
# Chart 9: Account-level best regime distribution
# ---------------------------------------------------------------
acct_pivot = pd.read_csv('E:\\Analysis_Support\\data_exports\\account_pnl_by_regime.csv', index_col=0)
counts = acct_pivot['best_regime'].value_counts().reindex(['Fear', 'Neutral', 'Greed']).fillna(0)
fig, ax = plt.subplots(figsize=(6, 4.5))
colors3 = ['#d9642c', '#9e9e9e', '#5a9c5f']
bars = ax.bar(counts.index, counts.values, color=colors3, width=0.55, zorder=3)
for b, v in zip(bars, counts.values):
    ax.text(b.get_x()+b.get_width()/2, v+0.3, f"{int(v)}", ha='center', fontsize=11, fontweight='bold')
style_ax(ax)
ax.set_ylabel('Number of accounts (of 32)')
ax.set_title('Which Sentiment Regime is Each Trader Most\nProfitable In?', fontsize=13, fontweight='bold', loc='left')
plt.tight_layout()
plt.savefig(f'{OUT}/09_account_best_regime.png', dpi=150)
plt.close()

print("All 9 charts saved to", OUT)
import os
for f in sorted(os.listdir(OUT)):
    print(' -', f)
