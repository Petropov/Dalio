#!/usr/bin/env python3
"""
dashboard.py — Cross-Asset Capital Rotation (Weekly)

What this script does (end-to-end):
1) Downloads market data proxies for each bucket with yfinance
2) Computes multi-horizon momentum and a normalized “share” per bucket
3) Adds WoW delta using last week’s cached snapshot (data/last_week.parquet)
4) Computes a headline Rotation Index (Risk-On → Risk-Off net flow, pp WoW)
5) Appends to risk_on_share history and draws a 4-week sparkline
6) Builds a compact heatmap (buckets × horizons) of z-scored momentum
7) Writes a single HTML report at ./report.html

No external API keys required. Internet access is required on first run.
"""

from __future__ import annotations
import io
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -----------------------------
# Config
# -----------------------------

OUT_HTML = Path("report.html")
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

LAST_WEEK_SNAPSHOT = DATA_DIR / "last_week.parquet"
RISKON_HISTORY_CSV = DATA_DIR / "risk_on_share.csv"

# Buckets and proxy tickers (keep simple, liquid ETFs / indices)
BUCKETS: Dict[str, List[str]] = {
    # Risk-ON
    "Equities (Global)": ["VT"],                       # Vanguard Total World
    "Credit / Carry":   ["LQD", "HYG"],                # IG & High Yield Credit
    "Commodities":      ["DBC"],                       # Broad commodities
    "Crypto / Spec":    ["BTC-USD"],                   # Bitcoin (spec proxy)

    # Risk-OFF / Defensive
    "Rates / Duration": ["IEF"],                       # 7–10Y UST (duration proxy)
    "Gold (Hedge)":     ["GLD"],
    "USD & FX":         ["UUP"],                       # US Dollar index ETF
    "Cash / Sidelines": ["BIL"],                       # 1–3M T-Bill
}

# Which buckets belong to risk-on vs risk-off for the Rotation Index
RISK_ON  = {"Equities (Global)", "Credit / Carry", "Commodities", "Crypto / Spec"}
RISK_OFF = {"Rates / Duration", "Gold (Hedge)", "USD & FX", "Cash / Sidelines"}

# Horizons (in trading days, ~ approximations)
HORIZONS = {
    "1m": 21,
    "3m": 63,
    "6m": 126,
    "12m": 252,
}

# Weighting for the “share” calculation (softmax across buckets)
SHARE_WEIGHTS = {"1m": 0.2, "3m": 0.3, "6m": 0.3, "12m": 0.2}

# -----------------------------
# Data helpers
# -----------------------------

def download_prices(tickers: List[str], start: str = "2019-01-01") -> pd.DataFrame:
    """Download adjusted close prices for a list of tickers into one DataFrame."""
    df = yf.download(tickers, start=start, auto_adjust=True, progress=False)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame()
    # Some crypto / fx may have missing weekends; forward-fill small gaps
    df = df.sort_index().ffill()
    return df

def combined_bucket_price(tickers: List[str]) -> pd.Series:
    """
    Combine multiple proxies into a single bucket series.
    Approach: equal-weighted normalized index rebased to 100.
    """
    prices = download_prices(tickers)
    # Rebase each series to 100 and average (equal weight)
    reindexed = prices / prices.iloc[0] * 100.0
    bucket = reindexed.mean(axis=1)
    bucket.name = "price"
    return bucket

def get_all_bucket_series() -> pd.DataFrame:
    """Return DataFrame with one column per bucket (equal-weighted proxy index)."""
    cols = {}
    for bucket, ticks in BUCKETS.items():
        cols[bucket] = combined_bucket_price(ticks)
    df = pd.DataFrame(cols).dropna(how="all")
    # Forward fill to align assets with uneven calendars
    df = df.ffill()
    return df

def multi_horizon_momentum(series: pd.Series, horizons: Dict[str, int]) -> pd.Series:
    """Compute simple total returns over given horizons."""
    out = {}
    for label, lookback in horizons.items():
        if len(series) <= lookback:
            out[label] = np.nan
        else:
            out[label] = (series.iloc[-1] / series.iloc[-lookback] - 1.0) * 100.0  # % return
    return pd.Series(out)

def zscore(x: pd.Series) -> pd.Series:
    mu, sigma = x.mean(), x.std(ddof=0)
    if sigma == 0 or np.isnan(sigma):
        return pd.Series(np.zeros_like(x), index=x.index)
    return (x - mu) / sigma

# -----------------------------
# Model the “share” and WoW
# -----------------------------

@dataclass
class BucketSnapshot:
    table: pd.DataFrame        # columns: ['share','WoW_pp','1m','3m','6m','12m']
    rotation_index_pp: float   # pp flow Risk-On -> Risk-Off
    risk_on_share: float       # current total risk-on %
    blurb: str                 # tiny auto commentary
    heatmap: pd.DataFrame      # z-scored momentum matrix (rows=buckets, cols=horizons)

def compute_snapshot() -> BucketSnapshot:
    prices = get_all_bucket_series()

    # Compute multi-horizon momentum (levels in %)
    mom = []
    for bucket in prices.columns:
        mom.append(multi_horizon_momentum(prices[bucket].dropna(), HORIZONS))
    mom = pd.DataFrame(mom, index=prices.columns)  # buckets × horizons (%)

    # Heatmap matrix: z-score across buckets, for each horizon
    heatmap = pd.DataFrame(
        {h: zscore(mom[h]) for h in HORIZONS.keys()},
        index=mom.index
    )

    # Build a single scalar “score” for share via weighted sum of horizon momentum z-scores
    score = pd.Series(0.0, index=heatmap.index)
    for h, w in SHARE_WEIGHTS.items():
        score += heatmap[h] * w

    # Softmax to convert to percentages that sum to 100
    exps = np.exp(score - score.max())  # numeric stability
    share = (exps / exps.sum() * 100.0).rename("share")

    # Load last week snapshot (if available) for WoW delta
    prev = pd.read_parquet(LAST_WEEK_SNAPSHOT) if LAST_WEEK_SNAPSHOT.exists() else None
    if prev is not None and "share" in prev.columns:
        wow_pp = (share - prev["share"]).rename("WoW_pp")
    else:
        wow_pp = pd.Series(np.nan, index=share.index, name="WoW_pp")

    # Rotation Index (pp): Risk-On outflow positive means into Risk-Off
    rotation_index_pp = float(
        wow_pp.loc[list(RISK_OFF)].sum(skipna=True) - wow_pp.loc[list(RISK_ON)].sum(skipna=True)
    )

    # Risk-on total share (for sparkline history)
    risk_on_share = float(share.loc[list(RISK_ON)].sum())

    # Tiny commentary: list top risers / fallers WoW
    if wow_pp.notna().any():
        top_rise = wow_pp.nlargest(2).dropna()
        top_fall = wow_pp.nsmallest(2).dropna()
        def fmt(items: pd.Series) -> str:
            names = ", ".join(items.index)
            vals = ", ".join(f"{v:+.1f}" for v in items.values)
            return f"{names} ({vals} pp)"
        blurb = f"Flows tilted out of {fmt(top_fall)} and into {fmt(top_rise)}."
    else:
        blurb = "First run — no WoW deltas yet."

    # Build table
    table = pd.concat([share, wow_pp, mom[["1m", "3m", "6m", "12m"]]], axis=1).reindex(share.sort_values(ascending=False).index)

    # Persist this snapshot for next run
    table.to_parquet(LAST_WEEK_SNAPSHOT)

    return BucketSnapshot(
        table=table,
        rotation_index_pp=rotation_index_pp,
        risk_on_share=risk_on_share,
        blurb=blurb,
        heatmap=heatmap.loc[table.index]  # same order as table
    )

# -----------------------------
# Charts
# -----------------------------

def fig_table(snapshot: BucketSnapshot) -> go.Figure:
    df = snapshot.table.copy()

    def arrow(v):
        if pd.isna(v): return ""
        return "▲" if v > 0 else ("▼" if v < 0 else "•")

    wow_with_arrows = df["WoW_pp"].apply(lambda v: f"{arrow(v)} {v:+.1f}" if pd.notna(v) else "—")
    share_fmt = df["share"].map(lambda v: f"{v:.1f}%")
    mom_cols = [df[c].map(lambda v: f"{v:+.1f}%") for c in ["1m","3m","6m","12m"]]

    header = ["Bucket", "Share", "WoW Δ (pp)", "1m", "3m", "6m", "12m"]
    cells  = [
        df.index.tolist(),
        share_fmt.tolist(),
        wow_with_arrows.tolist(),
        mom_cols[0].tolist(),
        mom_cols[1].tolist(),
        mom_cols[2].tolist(),
        mom_cols[3].tolist(),
    ]

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=header,
                    fill_color="lightgray",
                    align="left",
                    font=dict(size=12)
                ),
                cells=dict(
                    values=cells,
                    align="left",
                    height=26
                )
            )
        ]
    )
    fig.update_layout(margin=dict(l=10,r=10,t=10,b=10))
    return fig

def fig_heatmap(snapshot: BucketSnapshot) -> go.Figure:
    mat = snapshot.heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=mat.values,
            x=list(mat.columns),
            y=list(mat.index),
            coloraxis="coloraxis",
            zmin=-2.5, zmax=2.5
        )
    )
    fig.update_layout(
        coloraxis=dict(colorscale="RdBu", cmin=-2.5, cmax=2.5, reversescale=True),
        margin=dict(l=60,r=20,t=30,b=40),
        title="Relative Positioning (z-score of momentum)"
    )
    return fig

def fig_sparkline(history_csv: Path) -> go.Figure:
    if history_csv.exists():
        hist = pd.read_csv(history_csv, parse_dates=["date"])
        hist = hist.drop_duplicates("date").sort_values("date").tail(8)  # last ~8 obs (~wks)
    else:
        hist = pd.DataFrame({"date":[pd.Timestamp.today().date()], "share":[np.nan]})

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist["date"],
        y=hist["share"],
        mode="lines+markers",
        line=dict(width=2),
        marker=dict(size=5)
    ))
    fig.update_layout(
        height=160,
        margin=dict(l=30,r=10,t=10,b=30),
        yaxis_title="Risk-On Share (%)",
        title="Risk-On Share (last 4–8 obs)"
    )
    return fig

# -----------------------------
# HTML Assembly
# -----------------------------

def build_html(snapshot: BucketSnapshot) -> str:
    # Headline metrics
    headline = f"""
    <h2 style="margin:0">Cross-Asset Capital Rotation — Weekly</h2>
    <p style="margin:4px 0 0 0;color:#555">
      Rotation Index (Risk-On → Risk-Off, WoW): <b>{snapshot.rotation_index_pp:+.1f} pp</b><br/>
      Current Risk-On Share: <b>{snapshot.risk_on_share:.1f}%</b>
    </p>
    <p style="margin:6px 0 16px 0">{snapshot.blurb}</p>
    """

    # Plotly figs (render to HTML fragments)
    table_fig = fig_table(snapshot)
    heatmap_fig = fig_heatmap(snapshot)
    spark_fig = fig_sparkline(RISKON_HISTORY_CSV)

    table_html   = table_fig.to_html(include_plotlyjs="cdn", full_html=False)
    heatmap_html = heatmap_fig.to_html(include_plotlyjs=False, full_html=False)
    spark_html   = spark_fig.to_html(include_plotlyjs=False, full_html=False)

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Cross-Asset Capital Rotation — Weekly</title>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 24px; }}
    .grid {{ display: grid; grid-template-columns: 1fr; gap: 16px; }}
    @media (min-width: 1000px) {{
      .grid {{ grid-template-columns: 1.2fr 0.8fr; }}
    }}
    .card {{ border: 1px solid #eee; border-radius: 10px; padding: 16px; }}
    .muted {{ color:#666; font-size: 12px; }}
  </style>
</head>
<body>
  {headline}
  <div class="grid">
    <div class="card">
      <h3 style="margin-top:0">Bucket Snapshot</h3>
      {table_html}
      <p class="muted">Shares are normalized (softmax of momentum z-scores) so all buckets sum to 100%.</p>
    </div>
    <div class="card">
      <h3 style="margin-top:0">Short-Run Context</h3>
      {spark_html}
      <div style="height:8px"></div>
      <h3 style="margin:12px 0 6px 0">Rotation Heatmap</h3>
      {heatmap_html}
    </div>
  </div>
  <p class="muted">Generated: {pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M")} UTC</p>
</body>
</html>
"""
    return html

# -----------------------------
# Risk-on share history updater
# -----------------------------

def append_risk_on_history(value: float) -> None:
    if RISKON_HISTORY_CSV.exists():
        hist = pd.read_csv(RISKON_HISTORY_CSV)
    else:
        hist = pd.DataFrame(columns=["date","share"])
    today = pd.Timestamp.today().date()
    hist = pd.concat([hist, pd.DataFrame([{"date": today, "share": value}])], ignore_index=True)
    hist = hist.drop_duplicates("date", keep="last").sort_values("date")
    hist.to_csv(RISKON_HISTORY_CSV, index=False)

# -----------------------------
# Main
# -----------------------------

def main():
    snapshot = compute_snapshot()
    append_risk_on_history(snapshot.risk_on_share)
    html = build_html(snapshot)
    OUT_HTML.write_text(html, encoding="utf-8")
    print("✅ wrote report.html")

if __name__ == "__main__":
    main()
