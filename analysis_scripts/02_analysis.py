import pandas as pd
import numpy as np
from scipy import stats

pd.set_option('display.width', 160)
pd.set_option('display.max_columns', 20)

merged = pd.read_pickle('E:\\Analysis_Support\\merged.pkl')
merged = merged.dropna(subset=['sentiment']).copy()

sent_order5 = ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']
sent_order3 = ['Fear', 'Neutral', 'Greed']
merged['sentiment'] = pd.Categorical(merged['sentiment'], categories=sent_order5, ordered=True)
merged['sentiment_3'] = pd.Categorical(merged['sentiment_3'], categories=sent_order3, ordered=True)

# ============================================================
# 1. Overall trade-level stats by sentiment (5-class)
# ============================================================
def agg_block(df, groupcol):
    g = df.groupby(groupcol, observed=True)
    out = pd.DataFrame({
        'n_trades': g.size(),
        'n_accounts': g['account'].nunique(),
        'total_volume_usd': g['size_usd'].sum(),
        'avg_trade_size_usd': g['size_usd'].mean(),
        'total_closed_pnl': g['closed_pnl'].sum(),
        'total_fees': g['fee'].sum(),
        'total_net_pnl': g['net_pnl'].sum(),
        'avg_closed_pnl_per_trade': g['closed_pnl'].mean(),
        'median_closed_pnl_per_trade': g['closed_pnl'].median(),
    })
    # win rate computed only among trades that realized a non-zero PnL (i.e. closes)
    realized = df[df['has_realized_pnl']]
    gr = realized.groupby(groupcol, observed=True)
    out['n_realized_trades'] = gr.size()
    out['win_rate_pct'] = (gr.apply(lambda x: (x['closed_pnl'] > 0).mean(), include_groups=False) * 100)
    out['avg_win_usd'] = gr.apply(lambda x: x.loc[x['closed_pnl'] > 0, 'closed_pnl'].mean(), include_groups=False)
    out['avg_loss_usd'] = gr.apply(lambda x: x.loc[x['closed_pnl'] < 0, 'closed_pnl'].mean(), include_groups=False)
    out['pnl_per_$_volume_bps'] = (out['total_closed_pnl'] / out['total_volume_usd']) * 10000
    return out

by_sent5 = agg_block(merged, 'sentiment')
by_sent3 = agg_block(merged, 'sentiment_3')

print("="*100)
print("TRADE-LEVEL METRICS BY SENTIMENT (5-CLASS)")
print("="*100)
print(by_sent5.round(2).to_string())

print("\n" + "="*100)
print("TRADE-LEVEL METRICS BY SENTIMENT (3-CLASS)")
print("="*100)
print(by_sent3.round(2).to_string())

by_sent5.to_csv('E:\\Analysis_Support\\data_exports\\metrics_by_sentiment5.csv')
by_sent3.to_csv('E:\\Analysis_Support\\data_exports\\metrics_by_sentiment3.csv')

# ============================================================
# 2. Long vs Short bias by sentiment
# ============================================================
merged['is_long_open'] = merged['direction'] == 'Open Long'
merged['is_short_open'] = merged['direction'] == 'Open Short'
opens = merged[merged['direction'].isin(['Open Long', 'Open Short'])]
open_bias = opens.groupby('sentiment', observed=True)['direction'].value_counts(normalize=True).unstack() * 100
print("\n" + "="*100)
print("LONG vs SHORT OPEN-POSITION BIAS BY SENTIMENT (% of opens)")
print("="*100)
print(open_bias.round(1))
open_bias.to_csv('E:\\Analysis_Support\\data_exports\\long_short_bias.csv')

# ============================================================
# 3. Statistical significance: PnL differs across sentiment?
# ============================================================
realized = merged[merged['has_realized_pnl']]
groups = [realized.loc[realized['sentiment'] == s, 'closed_pnl'].values for s in sent_order5]
groups = [g for g in groups if len(g) > 1]
f_stat, p_val = stats.kruskal(*groups)
print("\n" + "="*100)
print("KRUSKAL-WALLIS TEST: does closed PnL distribution differ across the 5 sentiment classes?")
print(f"H-statistic = {f_stat:.2f}, p-value = {p_val:.6g}")
print("="*100)

# Fear vs Greed (extreme) direct comparison
fear_pnl = realized.loc[realized['sentiment'].isin(['Fear', 'Extreme Fear']), 'closed_pnl']
greed_pnl = realized.loc[realized['sentiment'].isin(['Greed', 'Extreme Greed']), 'closed_pnl']
u_stat, p_val2 = stats.mannwhitneyu(fear_pnl, greed_pnl, alternative='two-sided')
print(f"\nMann-Whitney U (Fear-side vs Greed-side trades): U={u_stat:.0f}, p={p_val2:.6g}")
print(f"Fear-side: n={len(fear_pnl)}, mean={fear_pnl.mean():.2f}, median={fear_pnl.median():.2f}, win_rate={(fear_pnl>0).mean()*100:.2f}%")
print(f"Greed-side: n={len(greed_pnl)}, mean={greed_pnl.mean():.2f}, median={greed_pnl.median():.2f}, win_rate={(greed_pnl>0).mean()*100:.2f}%")

# ============================================================
# 4. Trade size / risk appetite by sentiment
# ============================================================
size_stats = merged.groupby('sentiment', observed=True)['size_usd'].describe()
print("\n" + "="*100)
print("TRADE SIZE (USD) DISTRIBUTION BY SENTIMENT")
print("="*100)
print(size_stats.round(2))
size_stats.to_csv('E:\\Analysis_Support\\data_exports\\trade_size_by_sentiment.csv')

# ============================================================
# 5. Per-account behaviour: does each trader do better in fear or greed?
# ============================================================
acct_sent = merged[merged['has_realized_pnl']].groupby(['account', 'sentiment_3'], observed=True).agg(
    n_trades=('closed_pnl', 'size'),
    total_pnl=('closed_pnl', 'sum'),
    win_rate=('closed_pnl', lambda x: (x > 0).mean() * 100)
).reset_index()
acct_pivot = acct_sent.pivot(index='account', columns='sentiment_3', values='total_pnl').fillna(0)
acct_pivot['best_regime'] = acct_pivot.idxmax(axis=1)
print("\n" + "="*100)
print("HOW MANY ACCOUNTS ARE MOST PROFITABLE IN EACH SENTIMENT REGIME")
print("="*100)
print(acct_pivot['best_regime'].value_counts())
acct_pivot.to_csv('E:\\Analysis_Support\\data_exports\\account_pnl_by_regime.csv')

# ============================================================
# 6. Top coins traded, and their PnL by sentiment
# ============================================================
top_coins = merged['coin'].value_counts().head(8).index.tolist()
coin_sent = merged[merged['coin'].isin(top_coins) & merged['has_realized_pnl']].groupby(
    ['coin', 'sentiment_3'], observed=True)['closed_pnl'].agg(['sum', 'mean', 'count'])
print("\n" + "="*100)
print(f"TOP {len(top_coins)} COINS: PnL BY SENTIMENT")
print("="*100)
print(coin_sent.round(2))
coin_sent.to_csv('E:\\Analysis_Support\\coin_pnl_by_sentiment.csv')

# ============================================================
# 7. Extreme regimes deep dive
# ============================================================
extreme = merged[merged['sentiment'].isin(['Extreme Fear', 'Extreme Greed']) & merged['has_realized_pnl']]
ext_stats = extreme.groupby('sentiment', observed=True).agg(
    n_trades=('closed_pnl', 'size'),
    win_rate=('closed_pnl', lambda x: (x > 0).mean() * 100),
    mean_pnl=('closed_pnl', 'mean'),
    total_pnl=('closed_pnl', 'sum'),
    avg_size_usd=('size_usd', 'mean')
)
print("\n" + "="*100)
print("EXTREME FEAR vs EXTREME GREED DEEP DIVE")
print("="*100)
print(ext_stats.round(2))

print("\nDONE - all outputs saved to /home/claude/analysis/outputs/")
