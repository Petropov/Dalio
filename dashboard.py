#!/usr/bin/env python3
"""Generate the Cross-Asset Rotation dashboard report."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import yfinance as yf

from rotation_decision_engine import (
    CONVEXITY_MAP,
    SignalBundle,
    align_score,
    apply_crowding_penalty,
    arrow_for,
    compute_signals,
    crowding_zscore,
    recommendation_plan,
    regime_call,
    what_breaks_view,
)

# ---------- paths ----------
DATA_DIR = Path("data")
OUT_DIR = Path("output")
STATE_DIR = Path("state")

for directory in (DATA_DIR, OUT_DIR, STATE_DIR):
    directory.mkdir(exist_ok=True)

SNAPSHOT_PATH = STATE_DIR / "last_snapshot.json"
HISTORY_PATH = DATA_DIR / "risk_on_share.csv"

TODAY = dt.date.today()
AS_OF = TODAY.isoformat()

# ---------- buckets & horizons ----------
BUCKETS: list[tuple[str, str]] = [
    ("Equities (global stocks)", "ACWI"),
    ("Credit / Carry (corp & EM bonds)", "HYG"),
    ("Rates / Duration (govt bonds)", "IEF"),
    ("Commodities (energy & metals)", "GSG"),
    ("Gold (defensive hedge)", "GLD"),
    ("Crypto / Spec (BTC, ETH)", "BTC-USD"),
    ("USD & FX (US dollar & majors)", "UUP"),
    ("Cash/Sidelines (synthetic)", "BIL"),
]

RISK_ON = {
    "Equities (global stocks)",
    "Credit / Carry (corp & EM bonds)",
    "Commodities (energy & metals)",
    "Crypto / Spec (BTC, ETH)",
}

DEFENSIVE = {
    "Rates / Duration (govt bonds)",
    "Gold (defensive hedge)",
    "USD & FX (US dollar & majors)",
    "Cash/Sidelines (synthetic)",
}

CYCLICAL = {name for name, _ in BUCKETS} - DEFENSIVE

HORIZONS: Mapping[str, int] = {
    "4W": 20,
    "12W": 60,
    "6M": 126,
    "12M": 252,
    "2Y": 504,
    "3Y": 756,
}

# ---------- helpers ----------
def _last_close(series: pd.Series) -> float:
    series = series.dropna()
    return float(series.iloc[-1]) if not series.empty else float("nan")


def _ret(series: pd.Series, window: int) -> float:
    series = series.dropna()
    if len(series) <= window:
        return float("nan")
    latest = series.iloc[-1]
    prev = series.iloc[-1 - window]
    return float((latest / prev - 1.0) * 100.0)


def _nanmean(values: Iterable[float]) -> float:
    filtered = [v for v in values if not pd.isna(v)]
    return float("nan") if not filtered else float(np.mean(filtered))


def _ret_at_position(series: pd.Series, position: int, window: int) -> float:
    if position <= window:
        return float("nan")
    latest = series.iloc[position]
    prev = series.iloc[position - window]
    try:
        return float((latest / prev - 1.0) * 100.0)
    except Exception:
        return float("nan")


def _softmax_weights(momentums: list[float]) -> np.ndarray:
    scores = np.clip(np.nan_to_num(momentums, nan=0.0), 0.0, None)
    if len(scores) == 0:
        return np.array([])
    if np.allclose(scores, 0.0):
        return np.full(len(scores), 1.0 / len(scores))
    temperature = np.std(scores) + 1e-6
    exp_scores = np.exp(scores / temperature)
    return exp_scores / np.sum(exp_scores)


# ---------- download prices ----------
tickers = " ".join(ticker for _, ticker in BUCKETS)
raw_data = yf.download(
    tickers=tickers,
    period="5y",
    interval="1d",
    auto_adjust=True,
    group_by="ticker",
    threads=True,
    progress=False,
)


def _extract_close(data: pd.DataFrame, ticker: str) -> pd.Series:
    """Return a closing price series for *ticker* from the downloaded data."""
    if isinstance(data.columns, pd.MultiIndex):
        if ticker in data.columns.get_level_values(0):
            sub = data[ticker]
            if isinstance(sub, pd.Series):
                return sub
            for field in ("Close", "Adj Close", "close", "adjclose"):
                if field in sub.columns:
                    return sub[field]
            if not sub.empty:
                return sub.iloc[:, 0]
    else:
        if ticker in data.columns:
            return data[ticker]
        for field in ("Close", "Adj Close"):
            if field in data.columns:
                col = data[field]
                if isinstance(col, pd.DataFrame) and ticker in col.columns:
                    return col[ticker]
                if isinstance(col, pd.Series):
                    return col
    return pd.Series(dtype=float)


price_columns: dict[str, pd.Series] = {}
for bucket, ticker in BUCKETS:
    series = _extract_close(raw_data, ticker)
    price_columns[bucket] = series.rename(bucket)

price_df = pd.concat(price_columns.values(), axis=1).sort_index().ffill()


# ---------- current snapshot ----------
rows: list[dict[str, float | str]] = []
momentum_vector: list[float] = []
for bucket, ticker in BUCKETS:
    prices = price_df[bucket]
    current = _last_close(prices)
    returns = {label: _ret(prices, window) for label, window in HORIZONS.items()}

    short = _nanmean([returns["4W"], returns["12W"]])
    medium = _nanmean([returns["6M"], returns["12M"]])
    long_term = _nanmean([returns["2Y"], returns["3Y"]])

    long_penalty = 0.0 if pd.isna(long_term) else max(0.0, -long_term)
    momentum = _nanmean([short, medium, medium])
    if not pd.isna(momentum):
        momentum -= 0.25 * long_penalty
    momentum_vector.append(momentum if not pd.isna(momentum) else 0.0)

    rows.append(
        {
            "Bucket": bucket,
            "Ticker": ticker,
            "Momentum": momentum,
            **{key: (None if pd.isna(value) else round(value, 2)) for key, value in returns.items()},
            "CurrentClose": None if pd.isna(current) else round(current, 2),
        }
    )


df = pd.DataFrame(rows)
weights = _softmax_weights(momentum_vector)
df["Current"] = np.round(weights * 100.0, 2)
df = df.drop(columns=["Momentum", "CurrentClose"])

# export simple CSV for downstream use
export_cols = ["Bucket", "Ticker", "Current", *HORIZONS.keys()]
df[export_cols].to_csv(OUT_DIR / "buckets.csv", index=False)

# load previous snapshot (if any) and compute WoW deltas
prior: dict[str, float] = {}
if SNAPSHOT_PATH.exists():
    try:
        prior = json.loads(SNAPSHOT_PATH.read_text())
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"⚠️ Could not read prior snapshot: {exc}")


def _pp_delta(curr: float, prev: float | None) -> float:
    try:
        return round(float(curr) - float(prev), 2)
    except Exception:
        return float("nan")


df["WoW"] = [
    _pp_delta(current, prior.get(bucket))
    for bucket, current in zip(df["Bucket"], df["Current"], strict=True)
]

# Rotation index: DEFENSIVE − CYCLICAL (level and 1-week change)
def _sum_column(names: Iterable[str], column: str) -> float:
    data = pd.to_numeric(df.loc[df["Bucket"].isin(names), column], errors="coerce")
    return float(data.fillna(0).sum())


rotation_level = round(_sum_column(DEFENSIVE, "Current") - _sum_column(CYCLICAL, "Current"), 2)
rotation_wow = round(_sum_column(DEFENSIVE, "WoW") - _sum_column(CYCLICAL, "WoW"), 2)
risk_on_share = round(_sum_column(RISK_ON, "Current"), 2)

print(f"Rotation Index ⇒ level {rotation_level:+.2f} pp | Δ1w {rotation_wow:+.2f} pp")

# persist snapshot & history
SNAPSHOT_PATH.write_text(json.dumps({row.Bucket: float(row.Current) for _, row in df.iterrows()}))
print(f"✅ wrote {SNAPSHOT_PATH}")

history_row = pd.DataFrame(
    [
        {
            "date": AS_OF,
            "risk_on_pct": risk_on_share,
            "ri_level": rotation_level,
            "ri_wow": rotation_wow,
        }
    ]
)

if HISTORY_PATH.exists():
    history = pd.read_csv(HISTORY_PATH)
    history = pd.concat([history, history_row], ignore_index=True)
    history = history.sort_values("date").drop_duplicates("date", keep="last")
else:
    history = history_row

history.to_csv(HISTORY_PATH, index=False)


# ---------- synthetic weight history for crowding ----------

def compute_weight_history(prices: pd.DataFrame) -> dict[str, list[float]]:
    if prices.empty:
        return {bucket: [] for bucket, _ in BUCKETS}
    weekly = prices.resample("W-FRI").last().ffill().dropna(how="all")
    weekly = weekly.tail(200)
    weekly_windows = {label: max(1, window // 5) for label, window in HORIZONS.items()}

    history_weights: dict[str, list[float]] = {bucket: [] for bucket, _ in BUCKETS}
    for pos in range(len(weekly)):
        momentums: list[float] = []
        bucket_moms: dict[str, float] = {}
        for bucket, _ in BUCKETS:
            series = weekly[bucket].ffill()
            returns = {
                label: _ret_at_position(series, pos, weekly_windows[label]) for label in HORIZONS
            }
            short = _nanmean([returns["4W"], returns["12W"]])
            medium = _nanmean([returns["6M"], returns["12M"]])
            long_term = _nanmean([returns["2Y"], returns["3Y"]])
            long_penalty = 0.0 if pd.isna(long_term) else max(0.0, -long_term)
            momentum = _nanmean([short, medium, medium])
            if not pd.isna(momentum):
                momentum -= 0.25 * long_penalty
            bucket_moms[bucket] = momentum if not pd.isna(momentum) else 0.0
            momentums.append(bucket_moms[bucket])

        weights_row = _softmax_weights(momentums)
        for idx, (bucket, _) in enumerate(BUCKETS):
            if len(weights_row) > idx:
                history_weights[bucket].append(round(float(weights_row[idx] * 100.0), 4))
    return history_weights


weight_history = compute_weight_history(price_df)

# ---------- signal compression ----------
bundles: list[SignalBundle] = []
alignment_rows: list[str] = []
for _, record in df.iterrows():
    momentums = {key: record[key] for key in HORIZONS}
    st, mt, lt = compute_signals(momentums)
    align = align_score(st, mt, lt)
    arrow = arrow_for(align)
    history_weights = weight_history.get(record["Bucket"], [])
    crowd_z = crowding_zscore(float(record["Current"]), history_weights)
    effective, crowded = apply_crowding_penalty(align, crowd_z)

    bundles.append(
        SignalBundle(
            bucket=record["Bucket"],
            st=st,
            mt=mt,
            lt=lt,
            align=align,
            align_arrow=arrow,
            crowd_z=crowd_z,
            crowded=crowded,
            effective_score=effective,
            current_weight=float(record["Current"]),
        )
    )

    crowd_badge = "⚠️" if crowded else ""
    alignment_rows.append(
        f"""
      <tr>
        <td>{record['Bucket']}</td>
        <td class=\"num\">{record['Current']:.2f}%</td>
        <td class=\"num\">{arrow_for(st)}</td>
        <td class=\"num\">{arrow_for(mt)}</td>
        <td class=\"num\">{arrow_for(lt)}</td>
        <td class=\"num\">{align:.2f}</td>
        <td class=\"num\">{record['WoW']:+.2f} pp</td>
        <td class=\"num\">{'' if crowd_z is None else round(crowd_z,2)}</td>
        <td>{crowd_badge}</td>
      </tr>
    """
    )

recommendations = recommendation_plan(bundles)
regime, confidence = regime_call(bundles, CYCLICAL, DEFENSIVE)
breakers = what_breaks_view(regime)


# ---------- render single clean HTML ----------
def fmt_pct(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value:,.2f}%"


def sign_class(value: float | None) -> str:
    if value is None or pd.isna(value) or abs(value) < 1e-9:
        return "muted"
    return "pos" if value > 0 else "neg"


STYLE = """
<style>
  :root{--bg:#0b0d11;--card:#0f1420;--ink:#e6ecf2;--muted:#9fb0c2;--acc:#4cc9f0;--ok:#10b981;--warn:#f59e0b}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial}
  .wrap{max-width:1100px;margin:32px auto;padding:0 18px}
  header{display:flex;justify-content:space-between;align-items:center;margin-bottom:18px}
  h1{margin:0;font:700 28px/1.1 system-ui}
  .muted{color:var(--muted)}
  .grid{display:grid;grid-template-columns:repeat(12,1fr);gap:16px}
  .card{background:var(--card);border-radius:12px;padding:14px 16px;box-shadow:0 2px 16px rgba(0,0,0,.25)}
  .kpi{display:flex;align-items:baseline;gap:10px}
  .kpi .v{font:700 28px/1}
  .kpi .sub{color:var(--muted);font-size:12px}
  table{width:100%;border-collapse:collapse;margin-top:12px}
  th,td{padding:10px 8px;border-bottom:1px solid rgba(255,255,255,.1)}
  th{font:600 12px/1.2 system-ui;text-transform:uppercase;letter-spacing:.04em;color:#cdd6e0;text-align:left}
  td{vertical-align:middle}
  .num{text-align:right;font-variant-numeric:tabular-nums}
  .pos{color:var(--ok)}
  .neg{color:#ef4444}
  .footer{color:var(--muted);font-size:12px;margin-top:28px;text-align:center}
  .pill{display:inline-flex;align-items:center;gap:8px;padding:10px 12px;border-radius:12px;background:rgba(255,255,255,.05);margin-right:8px}
  .label{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
  .alert{color:var(--warn);font-weight:600}
  .list-compact{margin:0;padding-left:18px;color:var(--muted)}
  .list-compact li{margin-bottom:4px}
</style>
"""

risk_class = "pos" if risk_on_share > 50 else "neg"

recommendation_html = []
for bucket, delta in recommendations["INCREASE"]:
    recommendation_html.append(f"<div class='pill pos'><span class='label'>INCREASE</span>{bucket} (+{delta:.2f} pp)</div>")
for bucket, delta in recommendations["HOLD"]:
    recommendation_html.append(f"<div class='pill muted'><span class='label'>HOLD</span>{bucket} ({delta:+.2f} pp)</div>")
for bucket, delta in recommendations["REDUCE"]:
    recommendation_html.append(f"<div class='pill neg'><span class='label'>REDUCE</span>{bucket} ({delta:+.2f} pp)</div>")

crowded_assets = [b.bucket for b in bundles if b.crowded]
convexity_rows = []
for bucket, (downside, convex, role) in CONVEXITY_MAP.items():
    convexity_rows.append(
        f"<tr><td>{bucket}</td><td class='num'>{downside}</td><td class='num'>{convex}</td><td class='num'>{role}</td></tr>"
    )

SUMMARY = f"""
<header>
  <h1>Cross-Asset&nbsp;Capital&nbsp;Rotation <span class="muted">— Weekly</span></h1>
  <div class="muted">As of {AS_OF}</div>
</header>
<section class="card" style="margin-bottom:16px">
  <div class="kpi" style="justify-content:space-between;align-items:flex-start;gap:20px">
    <div>
      <div class="label">Regime call</div>
      <div class="v">{regime}</div>
      <div class="muted" style="margin-top:4px">Confidence {confidence:.1f}/10</div>
    </div>
    <div style="flex:1">
      <div class="label">Top actions</div>
      <div style="display:flex;flex-wrap:wrap;gap:8px">{''.join(recommendation_html)}</div>
    </div>
    <div>
      <div class="label">What breaks this view?</div>
      <ul class="list-compact">
        {''.join(f'<li>{item}</li>' for item in breakers)}
      </ul>
    </div>
  </div>
</section>
<section class="grid" style="margin-bottom:16px">
  <div class="card" style="grid-column: span 4">
    <div class="kpi">
      <div class="v {'pos' if rotation_level > 0 else ('neg' if rotation_level < 0 else 'muted')}">{rotation_level:+.2f} pp</div>
      <div class="sub">Rotation index (Defensive − Cyclical)</div>
    </div>
    <div class="{'pos' if rotation_wow > 0 else ('neg' if rotation_wow < 0 else 'muted')}" style="margin-top:6px">
      Δ 1-week: {rotation_wow:+.2f} pp
    </div>
  </div>
  <div class="card" style="grid-column: span 4">
    <div class="kpi">
      <div class="v {risk_class}">{risk_on_share:.2f}%</div>
      <div class="sub">Risk-On share (Equity/Credit/Commodities/Crypto)</div>
    </div>
  </div>
  <div class="card" style="grid-column: span 4">
    <div class="kpi">
      <div class="v">Crowding check</div>
    </div>
    <div class="muted" style="margin-top:6px">Z-score ≥ +1 flags crowded. Penalty applied to adds.</div>
    <div>{'None flagged' if not crowded_assets else ', '.join(crowded_assets)}</div>
  </div>
</section>
"""

TABLE = f"""
<section class="card">
  <div class="kpi" style="margin-bottom:8px">
    <div class="sub">3-signal alignment (ST/MT/LT) and crowding</div>
  </div>
  <table>
    <thead>
      <tr>
        <th>Bucket</th>
        <th class="num">Current %</th>
        <th class="num">ST</th>
        <th class="num">MT</th>
        <th class="num">LT</th>
        <th class="num">Align</th>
        <th class="num">Δ 1-w (pp)</th>
        <th class="num">Crowd z</th>
        <th></th>
      </tr>
    </thead>
    <tbody>
      {''.join(alignment_rows)}
    </tbody>
  </table>
</section>
"""

CONVEXITY = f"""
<section class="grid" style="margin-top:16px">
  <div class="card" style="grid-column: span 6">
    <div class="kpi"><div class="sub">Convexity map</div></div>
    <table>
      <thead><tr><th>Bucket</th><th class="num">Downside protected?</th><th class="num">Upside convex?</th><th class="num">Role</th></tr></thead>
      <tbody>{''.join(convexity_rows)}</tbody>
    </table>
  </div>
  <div class="card" style="grid-column: span 6">
    <div class="kpi"><div class="sub">Notes</div></div>
    <ul class="list-compact">
      <li>Signals: ST=0.6×4W+0.4×12W; MT=0.5×6M+0.5×12M; LT=0.5×2Y+0.5×3Y; tanh-normalized to [-1,+1].</li>
      <li>Alignment = 0.5×ST + 0.3×MT + 0.2×LT; arrows: ↑ if &gt;0.15, ↓ if &lt;-0.15 else →.</li>
      <li>Crowding penalty: if weight z ≥ +1.0, reduce positive alignment by 30%.</li>
      <li>Suggested sizing capped at ±1.5 pp with sum ≈ 0; held assets absorb residual.</li>
    </ul>
  </div>
</section>
"""

HTML = f"""<!doctype html>
<meta charset="utf-8">
{STYLE}
<div class="wrap">
  {SUMMARY}
  {TABLE}
  {CONVEXITY}
  <div class="footer">
    Generated {AS_OF}. Data: Yahoo Finance. “Current %” = softmax of blended momentum; snapshots in <code>{STATE_DIR}/</code>.
  </div>
</div>
"""

output_path = Path("report.html")
output_path.write_text(HTML, encoding="utf-8")
print("✅ wrote", output_path.resolve())
