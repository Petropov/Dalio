# dashboard.py
# Cross-Asset Capital Rotation — Weekly
# - Builds report.html with bucket table
# - Exports output/buckets.csv
# - Computes WoW deltas + Rotation Index and injects overlay
# - Persists state under state/

import os, json, math, datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf

# --- persistence ---
DATA_DIR = Path("data"); DATA_DIR.mkdir(exist_ok=True)
SNAP_FP = DATA_DIR / "last_week.parquet"
HIST_FP = DATA_DIR / "risk_on_share.csv"
OUT_DIR = Path("output")
STATE_DIR = Path("state")
OUT_DIR.mkdir(exist_ok=True, parents=True)
STATE_DIR.mkdir(exist_ok=True, parents=True)

TODAY = dt.date.today()

# ----------------------------
# 1) Buckets and proxy tickers
# ----------------------------
BUCKETS = [
    ("Equities (global stocks)", "ACWI"),
    ("Credit / Carry (corp & EM bonds)", "HYG"),
    ("Rates / Duration (govt bonds)", "IEF"),
    ("Commodities (energy & metals)", "GSG"),
    ("Gold (defensive hedge)", "GLD"),
    ("Crypto / Spec (BTC, ETH)", "BTC-USD"),   # BTC proxy
    ("USD & FX (US dollar & majors)", "UUP"),  # DXY proxy (ETF)
    ("Cash/Sidelines (synthetic)", "BIL"),     # 1-3m T-Bill ETF
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

# Horizons in trading days (approx)
HORIZONS = {
    "4W": 20,
    "12W": 60,
    "6M": 126,
    "12M": 252,
    "2Y": 504,
    "3Y": 756,
}

# ------------------------------------
# 2) Download prices and compute stats
# ------------------------------------
hist_days = max(HORIZONS.values()) + 10
tickers = [t for _, t in BUCKETS]
data = yf.download(
    tickers=tickers,
    period="5y",
    interval="1d",
    group_by="ticker",
    auto_adjust=True,
    progress=False,
    threads=True,
)

def last_close(series: pd.Series) -> float:
    s = series.dropna()
    return float(s.iloc[-1]) if len(s) else np.nan

def pct_return(series: pd.Series, window: int) -> float:
    s = series.dropna()
    if len(s) < window + 1:
        return np.nan
    return float((s.iloc[-1] / s.iloc[-1 - window] - 1.0) * 100.0)

rows = []
for bucket, ticker in BUCKETS:
    try:
        px = data[ticker]["Close"]
    except Exception:
        # In some yfinance formats, single level column
        px = data["Close"][ticker] if ("Close" in data and ticker in data["Close"].columns) else pd.Series(dtype=float)

    cur = last_close(px)
    ret = {k: pct_return(px, n) for k, n in HORIZONS.items()}

    # Composite momentum score: blend shorter horizons, lightly penalize long-term underperformance
    short = np.nanmean([ret["4W"], ret["12W"]])
    mid = np.nanmean([ret["6M"], ret["12M"]])
    long = np.nanmean([ret["2Y"], ret["3Y"]])
    comp = np.nanmean([short, mid, mid]) - 0.25 * (0 if np.isnan(long) else max(0, -long))

    # Momentum vs 12M label
    m_label = "Near"
    diff = (0 if np.isnan(ret["4W"]) or np.isnan(ret["12M"]) else ret["4W"] - ret["12M"])
    if diff > 0.4:
        m_label = "Higher"
    elif diff < -0.4:
        m_label = "Lower"

    row = {
        "Bucket": bucket,
        "Ticker": ticker,
        "Momentum (Now vs 12M)": m_label,
        "Current_score": comp,
        **ret,
    }
    rows.append(row)

df = pd.DataFrame(rows)

# -------------------------------------------------
# 3) Turn momentum into soft allocation (Current %)
# -------------------------------------------------
# Softmax over non-negative scores (clip at zero)
scores = np.clip(df["Current_score"].fillna(0.0).values, 0, None)
if np.all(scores == 0):
    # if all neutral, give everyone equal weight
    weights = np.ones_like(scores) / len(scores)
else:
    expw = np.exp(scores / (np.std(scores) + 1e-6))
    weights = expw / expw.sum()

current_pct = weights * 100.0
df.insert(df.columns.get_loc("Current_score") + 1, "Current", np.round(current_pct, 2))
df.drop(columns=["Current_score"], inplace=True)

# -------------------------------
# 4) Persist CSV for transparency
# -------------------------------
export_cols = ["Bucket", "Current"] + list(HORIZONS.keys())
df_export = df[export_cols].copy()
OUT_DIR.mkdir(exist_ok=True, parents=True)
df_export.to_csv(OUT_DIR / "buckets.csv", index=False)

# ---------------------------------------------
# 5) Compute WoW deltas + Rotation Index + save
# ---------------------------------------------
snap_path = STATE_DIR / "last_snapshot.json"
STATE_DIR.mkdir(exist_ok=True)

# --- Load previous snapshot (if any) ---
prior = {}
if snap_path.exists():
    try:
        prior = json.loads(snap_path.read_text())
    except Exception as e:
        print(f"⚠️ could not read prior snapshot: {e}")
        prior = {}

# --- Week-over-week change (pp) ---
df["WoW"] = df.apply(
    lambda r: (float(r["Current"]) - float(prior.get(r["Bucket"], np.nan)))
    if r["Bucket"] in prior
    else np.nan,
    axis=1,
)

# --- Rotation Index: defensive minus cyclical (levels & WoW) ---
lvl_def = df[df["Bucket"].isin(DEFENSIVE)]["Current"].sum()
lvl_cyc = df[df["Bucket"].isin(CYCLICAL)]["Current"].sum()
wow_def = df[df["Bucket"].isin(DEFENSIVE)]["WoW"].sum(skipna=True)
wow_cyc = df[df["Bucket"].isin(CYCLICAL)]["WoW"].sum(skipna=True)

rotation_index_lvl = lvl_def - lvl_cyc
rotation_index_wow = wow_def - wow_cyc

print(
    f"Rotation Index (levels): {rotation_index_lvl:+.2f} pp | "
    f"WoW change: {rotation_index_wow:+.2f} pp"
)

# --- Save snapshot for next run ---
snap_dict = {row["Bucket"]: float(row["Current"]) for _, row in df.iterrows()}
snap_path.write_text(json.dumps(snap_dict))
print(f"✅ wrote snapshot: {snap_path.resolve()}")

