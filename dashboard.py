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


rows: list[dict[str, float | str]] = []
for bucket, ticker in BUCKETS:
    prices = _extract_close(raw_data, ticker)
    current = _last_close(prices)
    returns = {label: _ret(prices, window) for label, window in HORIZONS.items()}

    short = _nanmean([returns["4W"], returns["12W"]])
    medium = _nanmean([returns["6M"], returns["12M"]])
    long_term = _nanmean([returns["2Y"], returns["3Y"]])

    long_penalty = 0.0 if pd.isna(long_term) else max(0.0, -long_term)
    momentum = _nanmean([short, medium, medium])
    if not pd.isna(momentum):
        momentum -= 0.25 * long_penalty

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

# softmax over non-negative scores
scores = np.clip(df["Momentum"].fillna(0.0).to_numpy(), 0.0, None)
if np.allclose(scores, 0.0):
    weights = np.full(len(scores), 1.0 / max(len(scores), 1))
else:
    temperature = np.std(scores) + 1e-6
    exp_scores = np.exp(scores / temperature)
    weights = exp_scores / np.sum(exp_scores)

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
</style>
"""

risk_class = "pos" if risk_on_share > 50 else "neg"
SUMMARY = f"""
<header>
  <h1>Cross-Asset&nbsp;Capital&nbsp;Rotation <span class="muted">— Weekly</span></h1>
  <div class="muted">As of {AS_OF}</div>
</header>
<section class="grid" style="margin-bottom:16px">
  <div class="card" style="grid-column: span 6">
    <div class="kpi">
      <div class="v {'pos' if rotation_level > 0 else ('neg' if rotation_level < 0 else 'muted')}">{rotation_level:+.2f} pp</div>
      <div class="sub">Rotation index (Defensive − Cyclical)</div>
    </div>
    <div class="{'pos' if rotation_wow > 0 else ('neg' if rotation_wow < 0 else 'muted')}" style="margin-top:6px">
      Δ 1-week: {rotation_wow:+.2f} pp
    </div>
  </div>
  <div class="card" style="grid-column: span 6">
    <div class="kpi">
      <div class="v {risk_class}">{risk_on_share:.2f}%</div>
      <div class="sub">Risk-On share (Equity/Credit/Commodities/Crypto)</div>
    </div>
  </div>
</section>
"""

ROWS_HTML = []
for _, record in df.sort_values("Current", ascending=False).iterrows():
    ROWS_HTML.append(
        f"""
      <tr>
        <td>{record['Bucket']}</td>
        <td class="num">{fmt_pct(record['Current'])}</td>
        <td class="num {sign_class(record['4W'])}">{fmt_pct(record['4W'])}</td>
        <td class="num {sign_class(record['12W'])}">{fmt_pct(record['12W'])}</td>
        <td class="num {sign_class(record['6M'])}">{fmt_pct(record['6M'])}</td>
        <td class="num {sign_class(record['12M'])}">{fmt_pct(record['12M'])}</td>
        <td class="num {sign_class(record['2Y'])}">{fmt_pct(record['2Y'])}</td>
        <td class="num {sign_class(record['3Y'])}">{fmt_pct(record['3Y'])}</td>
        <td class="num {sign_class(record['WoW'])}">{fmt_pct(record['WoW'])}</td>
      </tr>
    """
    )

TABLE = f"""
<section class="card">
  <div class="kpi" style="margin-bottom:8px">
    <div class="sub">Allocation &amp; Momentum</div>
  </div>
  <table>
    <thead>
      <tr>
        <th>Bucket</th>
        <th class="num">Current %</th>
        <th class="num">4W %</th>
        <th class="num">12W %</th>
        <th class="num">6M %</th>
        <th class="num">12M %</th>
        <th class="num">2Y %</th>
        <th class="num">3Y %</th>
        <th class="num">Δ 1-w (pp)</th>
      </tr>
    </thead>
    <tbody>
      {''.join(ROWS_HTML)}
    </tbody>
  </table>
</section>
"""

HTML = f"""<!doctype html>
<meta charset="utf-8">
{STYLE}
<div class="wrap">
  {SUMMARY}
  {TABLE}
  <div class="footer">
    Generated {AS_OF}. Data: Yahoo Finance. “Current %” = softmax of blended momentum; snapshots in <code>{STATE_DIR}/</code>.
  </div>
</div>
"""

output_path = Path("report.html")
output_path.write_text(HTML, encoding="utf-8")
print("✅ wrote", output_path.resolve())
