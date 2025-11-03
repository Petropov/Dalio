# dashboard.py
# --------------------------------------------
# Cross-Asset Capital Rotation — Weekly (HTML)
# --------------------------------------------
# - Pulls market prices with yfinance
# - Builds a momentum-based "Current %" allocation per bucket
# - Computes 1-week delta (WoW) from last snapshot (state/last_snapshot.json)
# - Computes Rotation Index = (DEFENSIVE % − CYCLICAL %) and its 1-week change
# - Appends Risk-On share to data/risk_on_share.csv for history
# - Exports output/buckets.csv and regenerates report.html (single, non-duplicated)
# --------------------------------------------

import os
import json
import math
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# ---------- paths ----------
DATA_DIR = Path("data");  DATA_DIR.mkdir(exist_ok=True)
OUT_DIR  = Path("output"); OUT_DIR.mkdir(exist_ok=True)
STATE_DIR = Path("state"); STATE_DIR.mkdir(exist_ok=True)

SNAP_FP = STATE_NOISE = STATE_DIR / "last_snapshot.json"
HIST_FP = DATA_DIR / "risk_on_share.csv"

TODAY = dt.date.today()
ASOF_STR = TODAY.isoformat()

# ---------- buckets & horizons ----------
BUCKETS = [
    ("Equities (global stocks)", "ACWI"),
    ("Credit / Carry (corp & EM bonds)", "HYG"),
    ("Rates / Duration (govt bonds)", "IEF"),
    ("Commodities (energy & metals)", "GSG"),
    ("Gold (defensive hedge)", "GLD"),
    ("Crypto / Spec (BTC, ETH)", "BTC-USD"),   # BTC proxy for liquid crypto beta
    ("USD & FX (US dollar & majors)", "UUP"),  # DXY proxy ETF
    ("Cash/Sidelines (synthetic)", "BIL"),     # 1–3m T-Bill ETF
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
# Everything in BUCKETS that isn’t defensive is treated as cyclicals:
CYCLICAL = {name for name, _ in BUCKETS} - DEFENSIVE

HORIZONS = {  # trading-day approximations
    "4W": 20,
    "12W": 60,
    "6M": 126,
    "12M": 252,
    "2Y": 504,
    "3Y": 756,
}

# ---------- price download ----------
tickers = [t for _, t in BUCKETS]
hist_days = max(HORIZONS.values()) + 10
data = yf.download(
    tickers=" ".join(tickers),
    period="5y",
    interval="1d",
    auto_adjust=True,
    group_by="ticker",
    threads=True,
    progress=False,
)

def _last_close(series: pd.Series) -> float:
    s = series.dropna()
    return float(s.iloc[-1]) if len(s) else float("nan")

def _pct_return(series: pd.Series, window: int) -> float:
    s = series.dropna()
    if len(s) <= window:
        return float("nan")
    return float((s.iloc[-1] / s.iloc[-1 - window] - 1.0) * 100.0)

rows = []
for bucket, ticker in BUCKETS:
    # handle yfinance’s two possible MultiIndex layouts
    if isinstance(data.columns, pd.MultiIndex):
        if (ticker, "Close") in data.columns:
            px = data[(ticker, "Close")]
        else:
            # sometimes Close is a column level 0 with tickers as columns
            px = data["Close"][ticker] if "Close" in data.columns.levels[0] else pd.Series(dtype=float)
    else:
        px = data[ticker] if ticker in data.columns else pd.Series(dtype=float)

    cur = _last_close(px)
    rets = {h: _pct_return(px, n) for h, n in HORIZONS.items()}

    short = np.nanmean([rets["4W"], rets["12W"]])
    mid   = np.nanmean([rets["6M"],  rets["12M"]])
    long  = np.nanmean([rets["2Y"],  rets["3Y"]])

    # simple composite momentum: overweight recent trend, mildly penalize bad long-term drift
    comp = np.nanmean([short, mid, mid]) - 0.25 * (0.0 if np.isnan(long) else max(0.0, -long))

    d = {"Bucket": bucket, "Ticker": ticker, "MomentumScore": comp, **{k: round(v, 2) if pd.notnull(v) else np.nan for k, v in rets.items()}}
    rows.append(d)

df = pd.DataFrame(rows)

# ---------- soft allocation from momentum (softmax over non-neg.) ----------
scores = np.clip(df["MomentumScore"].fillna(0.0).to_numpy(), 0.0, None)
if np.allclose(scores, 0.0):
    w = np.full(len(scores), 1.0 / max(len(scores), 1))
else:
    temp = np.std(scores) + 1e-6
    ex = np.exp(scores / temp)
    w = ex / np.sum(ex)

df["Current"] = np.round(w * 100.0, 2)
df = df.drop(columns=["MomentumScore"])

# ---------- export transparent CSV snapshot ----------
export_cols = ["Bucket", "Ticker", "Current"] + list(HORIZONS.keys())
df[export_cols].to_csv(OUT_DIR / "buckets.csv", index=False)

# ---------- load previous snapshot & compute WoW deltas ----------
prior = {}
if SNAP_FP.exists():
    try:
        prior = json.load(open(SNAP_FP, "r"))
    except Exception as e:
        print(f"⚠️ Could not read prior snapshot: {e}")

def _pp_delta(curr, prev):
    try:
        return round(float(curr) - float(prev), 2)
    except Exception:
        return float("nan")

df["WoW"] = [
    _p p_delta(cur, prior.get(bucket)) for cur, bucket in zip(df["Current"].to_list(), df["Bucket"].to_list())
]

# ---------- rotation index (level & 1-week change) ----------
lvl_def = float(df.loc[df["Bucket"].isin(DEFENSIVE), "Current"].sum())
lvl_cyc = float(df.loc[df["Bucket"].isin(CYCLICAL),  "Current"].sum())
wow_def = float(pd.to_numeric(df.loc[df["Bucket"].isin(DEFENSIVE), "WoW"], errors="coerce").fillna(0).sum())
wow_cyc = float(pd.to_numeric(df.loc[df["Bucket"].isin(CYCLICAL),  "WoW"], errors="coerce").fillna(0).sum())

rotation_level = round(lvl_def - lvl_cyc, 2)
rotation_wow   = round(wow_def - wow_cyc, 2)

print(f"Rotation Index ⇒ level {rotation_level:+.2f} pp | Δ1w {rotation_wow:+.2f} pp")

# ---------- persist new snapshot & risk-on history ----------
json.dump({r.Bucket: float(r.Current) for _, r in df.iterrows()}, open(SNAP_FP, "w"))
print(f"✅ wrote {SNAP_FP}")

risk_on_share = float(df.loc[df["Bucket"].isin(RISK_ON), "Current"].sum())
hist_row = pd.DataFrame([{"date": ASOF_STR, "risk_on_pct": round(risk_on_share, 2),
                          "ri_level": rotation_level, "ri_wow": rotation_wow}])
if HIST_FP.exists():
    hist = pd.read_csv(HIST_FP)
    hist = pd.concat([hist, hist_row], ignore_index=True)
    # de-dup by date, keep last
    hist = (hist.sort_values(["date"])
                 .drop_duplicates(subset=["date"], keep="last"))
else:
    hist = hist_row
hist.to_csv(HIST_FP, index=False)

# ---------- build single, non-duplicated HTML ----------
style = f"""
<style>
  :root {{ --bg:#0b0d11; --card:#0f1420; --ink:#e6ecf2; --muted:#9fb0c2; --accent:#4cc9f0; --ok:#0ea5e9; --warn:#f59e0b; --buy:#10b981; --def:#94a3b8; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; font: 15px/1.55 system-ui, -apple-system, Segoe UI, Roboto, "Helvetica Neue", Arial, sans-serif; color:var(--ink); background:var(--bg); }}
  .wrap {{ max-width:1100px; margin: 32px auto; padding: 0 18px; }}
  header {{ display:flex; align-items:center; justify-content:space-between; margin-bottom:18px; }}
  h1 {{ font-size:28px; margin:0; font-weight:700; letter-spacing:.2px; }}
  .asof {{ color:var(--muted); font-size:13px; }}
  .pill {{ padding:.25rem .6rem; border-radius:999px; font-weight:600; font-size:12px; background:var(--def); color:#0b0d11; }}
  .pill.green {{ background:var(--buy); color:#062; }}
  .pill.orange {{ background:#f59e0b; color:#1a1000; }}
  .grid {{ display:grid; grid-template-columns: repeat(12, 1fr); gap:16px; }}
  .card {{ background:var(--card); border-radius:12px; padding:14px 16px; box-shadow:0 2px 16px rgba(0,0,0,.25); }}
  .kpi {{ display:flex; align-items:baseline; gap:10px; }}
  .kpi .v {{ font-size:28px; font-weight:700; }}
  .kpi .sub {{ color:var(--muted); font-size:12px; }}
  table {{ width:100%; border-collapse: collapse; margin-top: 12px; }}
  th, td {{ padding:10px 8px; text-align:left; border-bottom:1px solid rgba(255,255,255,.08); }}
  th {{ color:#cdd6e0; font-weight:600; text-transform:uppercase; font-size:12px; letter-spacing:.04em; }}
  td {{ vertical-align: middle; }}
  .num {{ text-align:right; font-variant-numeric: tabular-nums; }}
  .pos {{ color: var(--buy); }}
  .neg {{ color: #ef4444; }}
  .muted {{ color: var(--muted); }}
  .footer {{ color:var(--muted); font-size:12px; margin-top:28px; text-align:center; }}
</style>
"""

def fmt_pct(x):
    if pd.isna(x): return "—"
    return f"{x:,.2f}%"

def sign_cls(x):
    if pd.isna(x) or abs(x) < 1e-9: return "muted"
    return "pos" if x > 0 else "neg"

# single, non-duplicated header
header_html = f"""
<header>
  <h1>Cross-Asset&nbsp;Capital&nbsp;Rotation — <span class="muted">Weekly</span></h1>
  <div class="asof">As of {ASOF_STR}</div>
</header>
"""

# compact rotation summary (no duplicate big header)
rotation_html = f"""
<section class="grid" style="margin-bottom:16px">
  <div class="card" style="grid-column: span 6">
    <div class="kpi">
      <div class="v { 'pos' if rotation_level>0 else ('neg' if rotation_level<0 else 'muted') }">{rotation_level:+.2f} pp</div>
      <div class="sub">Rotation index (Defensive − Cyclical)</div>
    </div>
    <div class="{ 'pos' if rotation_wow>0 else ('neg' if rotation_wow<0 else 'muted') }" style="margin-top:6px;">
      Δ 1-week: {rotation_wow:+.2f} pp
    </div>
  </div>
  <div class="card" style="grid-column: span 6">
    <div class="kpi">
      <div class="v { 'pos' if risk_open:=risk_on_share>50 else 'neg' }">{risk_on_share:0.2f}%</div>
      <div class="sub">Risk-On share (Equity/Credit/Commodities/Crypto)</div>
    </div>
  </div>
</section>
""".replace("risk_open:=risk_on_share>50","pos" if risk_on_share>50 else "neg")

# one table with WoW column
show_cols = ["Bucket","Current","4W","12W","6M","12M","2Y","3Y","WoW"]
tbl_rows = []
for _, r in df.sort_values("Current", ascending=False).iterrows():
    tbl_rows.append(f"""
      <tr>
        <td>{r['Bucket']}</td>
        <td class="num">{fmt(R:=r['Current'])}</td>
        <td class="num {sign_cls(r['4W'])}">{fmt(r['4W'])}</td>
        <td class="num {sign_cls(r['12W'])}">{fmt(r['12M'] if pd.isna(r['12W']) else r['12W'])}</td>
        <td class="num {sign_cls(r['6M'])}">{fmt(r['6M'])}</td>
        <td class="num {sign_cls(r['12M'])}">{fmt(r['12M'])}</td>
        <td class="num {sign_cls(r['2Y'])}">{fmt(r['2Y'])}</td>
        <td class="num {sign_cls(r['3Y'])}">{fmt(r['3Y'])}</td>
        <td class="num {sign_cls(r['WoW'])}">{fmt(r['WoW'])}</td>
      </tr>
    """.replace("fmt(", "fmt_pct("))

table_html = f"""
<section class="card">
  <div class="kpi" style="margin-bottom:8px"><div class="sub">Allocation & Momentum</div></div>
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
      {''.join(tbl_rows)}
    </tbody>
  </table>
</section>
"""

page = f"""<!doctype html>
<meta charset="utf-8">
{style}
<div class="wrap">
  {header_html}
  {rotation_html}
  {table_html}
  <div class="footer">
    Generated {ASOF_STR}. Data: Yahoo Finance. “Current %” is a softmax-weighted momentum blend. 
    Snapshot history persisted under <code>{STATE_DIR}/</code>.
  </div>
</div>
"""

Path("report.html").write_text(page, encoding="utf-8")
print("✅ wrote report.html →", (Path("report.html").resolve()))
