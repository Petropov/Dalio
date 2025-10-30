
cat > dashboard.py << 'EOF'
# (paste the entire dashboard.py from my previous message here)

# dashboard.py
import yfinance as yf
import pandas as pd, numpy as np
from scipy.stats import trim_mean
from pandas_datareader import data as pdr
import time, math

# -----------------------------
# 0) Params & helpers
# -----------------------------
START = "2018-01-01"
INTERVAL = "1wk"
ATR_WIN = 4
BASELINE_WIN = 52
VOL_WIN = 8
MOM_COMPARE = "12M"          # or "6M"
RISK_ON_BASELINE = 35.0      # % used in narrative baseline

def hasnum(v):
    return (v is not None) and (not (isinstance(v,(float,np.floating)) and np.isnan(v)))

def fmt(v, unit=""):
    return f"{v:.2f}{unit}" if hasnum(v) else "—"

def fmt_chg3m(v, unit="pp"):
    if not hasnum(v): return ""
    sign = "▲" if v > 0 else ("▼" if v < 0 else "→")
    return f" <span style='color:#6b7280'>({sign} {abs(v):.2f}{unit} / 3m)</span>"

def fmt_z(v):
    return f" <span style='color:#6b7280'>z={v:+.2f}</span>" if hasnum(v) else ""

def badge(text, kind):
    colors = {"tight":"#ef4444","loose":"#16a34a","neutral":"#6b7280","inv":"#ef4444","steep":"#16a34a"}
    return f'<span style="background:{colors.get(kind,"#6b7280")};color:#fff;padding:2px 6px;border-radius:999px;font-size:12px;font-weight:600">{text}</span>'

def get_fred_series(code, start):
    # If you have a FRED API key, set env FRED_API_KEY and pandas_datareader will use it.
    for _ in range(3):
        try:
            s = pdr.DataReader(code, "fred", start=start).squeeze()
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:,0]
            return s.dropna()
        except Exception:
            time.sleep(1.2)
    return pd.Series(dtype=float)

# -----------------------------
# 1) Market universe (tickers)
# -----------------------------
tickers = [
    "UUP","CNYUSD=X","EURUSD=X",
    "TLT","BNDX",
    "HYG","EMB",
    "SPY","EEM","EZU",
    "DBC","CL=F",
    "GLD",
    "BTC-USD","ETH-USD"
]

BUCKET_EQ   = "Equities (global stocks)"
BUCKET_CR   = "Credit / Carry (corp & EM bonds)"
BUCKET_RT   = "Rates / Duration (govt bonds)"
BUCKET_CMD  = "Commodities (energy & metals)"
BUCKET_GLD  = "Gold (defensive hedge)"
BUCKET_CRY  = "Crypto / Spec (BTC, ETH)"
BUCKET_FX   = "USD & FX (US dollar & majors)"
BUCKET_CASH = "Cash/Sidelines (synthetic)"

bucket_map = {
    "UUP": BUCKET_FX, "CNYUSD=X": BUCKET_FX, "EURUSD=X": BUCKET_FX,
    "TLT": BUCKET_RT, "BNDX": BUCKET_RT,
    "HYG": BUCKET_CR, "EMB": BUCKET_CR,
    "SPY": BUCKET_EQ, "EEM": BUCKET_EQ, "EZU": BUCKET_EQ,
    "DBC": BUCKET_CMD, "CL=F": BUCKET_CMD,
    "GLD": BUCKET_GLD,
    "BTC-USD": BUCKET_CRY, "ETH-USD": BUCKET_CRY,
}
cash_label = BUCKET_CASH

# -----------------------------
# 2) Prices (Yahoo, weekly)
# -----------------------------
raw = yf.download(
    tickers, start=START, interval=INTERVAL,
    auto_adjust=True, group_by="ticker", progress=False, threads=True
)

def extract_price_panels(df, tks):
    closes, highs, lows = [], [], []
    for t in tks:
        sub = df[t]
        closes.append(sub["Close"].rename(t))
        highs.append(sub["High"].rename(t))
        lows.append(sub["Low"].rename(t))
    close = pd.concat(closes, axis=1)
    high  = pd.concat(highs,  axis=1).reindex(close.index)
    low   = pd.concat(lows,   axis=1).reindex(close.index)
    keep = close.columns[~close.isna().all()]
    close = close[keep].ffill().dropna(how="all")
    high  = high[keep].ffill()
    low   = low[keep].ffill()
    return close, high, low

close, high, low = extract_price_panels(raw, tickers)
rets = close.pct_change()

# -----------------------------
# 3) Activity × Stability score
# -----------------------------
atr_pct = ((high - low) / close).rolling(ATR_WIN, min_periods=max(2, ATR_WIN//2)).mean()

def trimmed_baseline(s, w=BASELINE_WIN, p=0.10):
    return s.rolling(w, min_periods=max(12, w//4)).apply(lambda x: trim_mean(x, proportiontocut=p), raw=False)

atr_base = atr_pct.copy()
for c in atr_pct.columns:
    atr_base[c] = trimmed_baseline(atr_pct[c])

activity_mult = (atr_pct / atr_base).clip(0.25, 4.0)
activity_weight = activity_mult.div(activity_mult.sum(1), axis=0)

vol8 = rets.rolling(VOL_WIN, min_periods=max(4, VOL_WIN//2)).std()
inv_vol = 1.0 / vol8.replace(0, np.nan)
inv_vol_norm = inv_vol.div(inv_vol.median(axis=1), axis=0)

idx = activity_weight.index.intersection(inv_vol_norm.index)
activity_weight = activity_weight.loc[idx].fillna(0.0)
inv_vol_norm = inv_vol_norm.loc[idx].ffill().fillna(1.0)
score = (activity_weight * inv_vol_norm).replace([np.inf, -np.inf], np.nan).fillna(0.0)

sum_score = score.sum(1)
risk_share = score.div((1.0 + sum_score), axis=0)
cash_share = (1.0 / (1.0 + sum_score)).rename(cash_label)

bucketed = risk_share.rename(columns=bucket_map)
bucketed = bucketed.T.groupby(level=0).sum().T
bucketed[cash_label] = cash_share

# -----------------------------
# 4) Macro drivers (FRED robust) + derived dials
# -----------------------------
fred_series = {
    "FEDFUNDS":"Policy rate (Fed Funds)",
    "DFII10":"Real 10Y yield (TIPS)",
    "T10YIE":"10Y inflation breakeven",
    "M2SL":"M2 (money supply)",
    "NFCI":"Financial conditions (NFCI)",
    "TOTCI":"C&I loans",
    "PCEPILFE":"Core PCE YoY (%)",
    "T10Y2Y":"2s10s yield curve (10Y-2Y, %)"
}
fred = {code: get_fred_series(code, START) for code in fred_series}
fred = pd.DataFrame({k:v for k,v in fred.items() if len(v)}).sort_index()
for k in fred_series:
    if k not in fred: fred[k] = np.nan
fred = fred.sort_index()

fred["M2SL_yoy"]   = fred["M2SL"].pct_change(52, fill_method=None) * 100
fred["TOTCI_yoy"]  = fred["TOTCI"].pct_change(52, fill_method=None) * 100
fred["FED_3m_bps"] = fred["FEDFUNDS"].diff(13) * 100
fred["core_pce_yoy"]   = fred["PCEPILFE"]
fred["real_policy"]    = fred["FEDFUNDS"] - fred["core_pce_yoy"]
fred["slope_2s10s"]    = fred["T10Y2Y"]
fred["NFCI_3m"]        = fred["NFCI"].diff(13)
fred["real_policy_3m"] = fred["real_policy"].diff(13)
fred["slope_2s10s_3m"] = fred["slope_2s10s"].diff(13)

def zscore(s, win=520, minp=260):
    m = s.rolling(win, min_periods=minp).mean()
    sd = s.rolling(win, min_periods=minp).std()
    return (s - m) / sd

z_last = {
    "real_policy_z": zscore(fred["real_policy"]).tail(1).iloc[0],
    "NFCI_z":        zscore(fred["NFCI"]).tail(1).iloc[0],
    "slope_2s10s_z": zscore(fred["slope_2s10s"]).tail(1).iloc[0],
    "DFII10_z":      zscore(fred["DFII10"]).tail(1).iloc[0],
    "T10YIE_z":      zscore(fred["T10YIE"]).tail(1).iloc[0],
    "M2SL_yoy_z":    zscore(fred["M2SL_yoy"]).tail(1).iloc[0],
    "TOTCI_yoy_z":   zscore(fred["TOTCI_yoy"]).tail(1).iloc[0],
}

now = pd.Timestamp.utcnow().tz_localize(None)
STALE_DAYS = {"FEDFUNDS":14,"NFCI":14,"DFII10":7,"T10YIE":7,"T10Y2Y":7,"PCEPILFE":60,"M2SL":60,"TOTCI":28}

def last_valid(series: pd.Series):
    s = series.dropna()
    if s.empty: return np.nan, None, True
    d = pd.to_datetime(s.index[-1])
    return float(s.iloc[-1]), d, False

LV = {}
for code in fred_series:
    v, d, _ = last_valid(fred[code]) if code in fred else (np.nan, None, True)
    stale = (d is None) or ((now - d).days > STALE_DAYS.get(code, 30))
    LV[code] = {"v": v, "date": d, "stale": stale}

def asof_tag(d, stale):
    if d is None: return ""
    color = "#ef4444" if stale else "#9ca3af"
    return f" <span style='color:{color}; font-size:11px'>(as of {pd.to_datetime(d).date()})</span>"

rp = (LV["FEDFUNDS"]["v"] - LV["PCEPILFE"]["v"]) if hasnum(LV["FEDFUNDS"]["v"]) and hasnum(LV["PCEPILFE"]["v"]) else np.nan
nf = LV["NFCI"]["v"]
sl = LV["T10Y2Y"]["v"]

def delta_3m(s: pd.Series):
    s = s.dropna()
    if len(s) < 14: return np.nan
    return float(s.iloc[-1] - s.iloc[-14])

rp_3m = delta_3m(fred["real_policy"])
nf_3m = delta_3m(fred["NFCI"])
sl_3m = delta_3m(fred["slope_2s10s"])

rp_v = "tight" if hasnum(rp) and rp>0 else ("loose" if hasnum(rp) and rp<0 else "neutral")
nf_v = "tight" if hasnum(nf) and nf>0 else ("loose" if hasnum(nf) and nf<-0.5 else "neutral")
sl_v = "inv" if hasnum(sl) and sl<0 else "steep"

# -----------------------------
# 5) Horizon table + Momentum (Now vs 12M)
# -----------------------------
windows = {"Current":1,"4W":4,"12W":12,"6M":26,"12M":52,"2Y":104,"3Y":156,"4Y":208}
def trailing_table(df, windows):
    cols = {}
    for label, w in windows.items():
        cols[label] = df.tail(1).T.squeeze() if w==1 else df.rolling(w, min_periods=max(4,w//4)).mean().iloc[-1]
    return (pd.concat(cols, axis=1) * 100).round(2)

table = trailing_table(bucketed, windows)
row_order = [BUCKET_EQ,BUCKET_CR,BUCKET_RT,BUCKET_CMD,BUCKET_GLD,BUCKET_CRY,BUCKET_FX,BUCKET_CASH]
table = table.reindex([r for r in row_order if r in table.index])
ref_series  = table[MOM_COMPARE]
curr_series = table["Current"]

def momentum_cell(current_pct, ref_pct):
    delta = current_pct - ref_pct
    if   delta > 0.2:  msg, color = "Higher", "#16a34a"
    elif delta < -0.2: msg, color = "Lower",  "#ef4444"
    else:              msg, color = "Near",   "#6b7280"
    txt = f'+{delta:.1f}pp' if delta >= 0 else f'−{abs(delta):.1f}pp'
    return f"<div class='mom'><span class='pill' style='background:{color}'>{msg}</span><span class='delta'>{txt}</span></div>"

momentum_cells = [momentum_cell(float(curr_series[b]), float(ref_series[b])) for b in table.index]
table_display = table.copy()
table_display.insert(0, "Momentum (Now vs "+MOM_COMPARE+")", momentum_cells)
table_display.insert(0, "Bucket", table_display.index)

def last_non_nan(series):
    if series is None:
        return np.nan
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(s.iloc[-1]) if len(s) else np.nan

# -----------------------------
# 6) Narrative tying flows to drivers
# -----------------------------
date_str = bucketed.index[-1].date().isoformat()
risk_on_names = [BUCKET_EQ,BUCKET_CR,BUCKET_CMD,BUCKET_CRY]
risk_off_names= [BUCKET_RT,BUCKET_GLD,BUCKET_FX,BUCKET_CASH]
rn, ro = curr_series[risk_on_names].sum(), curr_series[risk_off_names].sum()
rn_ref, ro_ref = ref_series[risk_on_names].sum(), ref_series[risk_off_names].sum()
tilt = rn - ro - (rn_ref - ro_ref)
drivers_flags = []
if rp_v=="tight": drivers_flags.append("real policy restrictive")
if nf_v=="tight": drivers_flags.append("financial conditions tight")
if sl_v=="inv":   drivers_flags.append("curve inverted")
if not drivers_flags: drivers_flags.append("drivers mixed")

# -----------------------------
# 7) Render dashboard (styles + layout)
# -----------------------------
def tint_cell(val, ref):
    if pd.isna(val) or pd.isna(ref): return ""
    diff = float(val) - float(ref)
    if diff > 0.5:  return 'style="background:#eaf8ed;"'
    if diff < -0.5: return 'style="background:#fdecec;"'
    return ""

styles = """
<style>
.rot-wrap { max-width:1320px; margin:10px auto; padding:0 12px;
            font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial; color:#000; }
.hdr { display:flex; justify-content:space-between; align-items:baseline; margin:6px 4px 10px 4px; color:#000; }
.hdr h2 { margin:0; font-size:18px; color:#000; }
.badge { padding:3px 8px; border-radius:999px; font-size:12px; color:#fff; }
.badge.TIGHTENING { background:#ef4444; } .badge.NEUTRAL { background:#6b7280; } .badge.EASING { background:#16a34a; }
.drivers { display:grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap:12px; margin:10px 2px 16px 2px; color:#000; }
.card { background:#fff; border:1px solid #e5e7eb; border-radius:10px; padding:10px 12px;
        box-shadow:0 1px 2px rgba(0,0,0,0.04); font-size:12.5px; color:#000; }
.card h4 { margin:0 0 6px 0; font-size:12.8px; color:#000; }
.kv { display:flex; justify-content:space-between; gap:10px; margin:2px 0; color:#000; }
.kv span:first-child { color:#000; }
.kv span:last-child { text-align:right; min-width:180px; }
.rot-table { border-collapse: collapse; width:100%; background:#fff; color:#000; font-size:13px;
             box-shadow:0 1px 3px rgba(0,0,0,0.08); table-layout:fixed; }
.rot-table th, .rot-table td { border:1px solid #d1d5db; padding:8px 10px; text-align:center; vertical-align:middle;
                               overflow:hidden; text-overflow:ellipsis; white-space:nowrap; color:#000; }
.rot-table thead th { background:#111; color:#fff; font-weight:700; }
.rot-table tbody tr:nth-child(even) { background:#f9fafb; }
.rot-table tbody tr:hover { background:#f3f4f6; }
.col-bucket { width: 260px; text-align:left !important; }
.col-momentum { width: 210px; overflow:visible !important; }
.mom { display:flex; align-items:center; gap:8px; justify-content:flex-start; white-space:nowrap; }
.pill { color:#fff; font-weight:600; font-size:12px; padding:1px 8px; border-radius:999px; }
.delta { color:#111; font-size:12px; }
.note { background:#fff; border:1px solid #e5e7eb; border-radius:8px; padding:12px 14px; margin:10px 0;
        font-size:14px; color:#000; }
.scroll-x { overflow-x:auto; -webkit-overflow-scrolling: touch; }
.map-table { width:100%; border-collapse:collapse; background:#fff; }
.map-table th, .map-table td { border:1px solid #e5e7eb; padding:8px 10px; text-align:left; white-space:nowrap; }
.foot { font-size:12.5px; color:#000; line-height:1.55; margin-top:10px; background:#fff;
        padding:10px 12px; border:1px solid #e5e7eb; border-radius:8px; }
</style>
"""

if rp_v == "loose" and nf_v == "loose":
    badge_color, badge_text, badge_class = "#16a34a", "EASING BIAS", "EASING"
elif rp_v == "tight" or nf_v == "tight":
    badge_color, badge_text, badge_class = "#ef4444", "TIGHTENING BIAS", "TIGHTENING"
else:
    badge_color, badge_text, badge_class = "#6b7280", "NEUTRAL / TRANSITION", "NEUTRAL"
hdr_badge = f'<span class="badge {badge_class}" style="background:{badge_color}">{badge_text}</span>'
date_str = bucketed.index[-1].date().isoformat()
hdr_html = f"<div class='hdr'><h2>Cross-Asset Capital Rotation — Weekly ({date_str})</h2>{hdr_badge}</div>"

def last_non_nan_safe(col):
    try:
        return last_non_nan(col)
    except Exception:
        return np.nan

m2_yoy_val    = last_non_nan_safe(fred.get("M2SL_yoy"))
totci_yoy_val = last_non_nan_safe(fred.get("TOTCI_yoy"))

drivers_html = f"""
<div class="drivers">
  <div class="card">
    <h4>Policy & Conditions</h4>
    <div class="kv"><span>Fed funds (%, level)</span><span>{fmt(LV['FEDFUNDS']['v'])}{asof_tag(LV['FEDFUNDS']['date'], LV['FEDFUNDS']['stale'])}</span></div>
    <div class="kv"><span>Real policy (FF − core PCE)</span><span>{fmt(rp,'%')} {badge('Restrictive','tight') if rp_v=='tight' else (badge('Accommodative','loose') if rp_v=='loose' else badge('Neutral','neutral'))}{fmt_chg3m(rp_3m,'pp')}</span></div>
    <div class="kv"><span>NFCI</span><span>{fmt(nf)} {badge('Tight','tight') if nf_v=='tight' else (badge('Loose','loose') if nf_v=='loose' else badge('Neutral','neutral'))}{fmt_chg3m(nf_3m,'')}{asof_tag(LV['NFCI']['date'], LV['NFCI']['stale'])}</span></div>
  </div>
  <div class="card">
    <h4>Real Rates & Inflation</h4>
    <div class="kv"><span>Real 10Y (TIPS)</span><span>{fmt(LV['DFII10']['v'],'%')}{fmt_z(z_last.get('DFII10_z'))}{asof_tag(LV['DFII10']['date'], LV['DFII10']['stale'])}</span></div>
    <div class="kv"><span>10Y breakeven</span><span>{fmt(LV['T10YIE']['v'],'%')}{fmt_z(z_last.get('T10YIE_z'))}{asof_tag(LV['T10YIE']['date'], LV['T10YIE']['stale'])}</span></div>
    <div class="kv"><span>2s10s slope</span><span>{fmt(sl,'%')} {badge('Inverted','inv') if sl_v=='inv' else badge('Steepening','steep')}{fmt_chg3m(sl_3m,'pp')}{asof_tag(LV['T10Y2Y']['date'], LV['T10Y2Y']['stale'])}</span></div>
  </div>
  <div class="card">
    <h4>Money & Credit</h4>
    <div class="kv"><span>M2 YoY</span><span>{fmt(m2_yoy_val,'%')}{fmt_z(z_last.get('M2SL_yoy_z'))}{asof_tag(LV['M2SL']['date'], LV['M2SL']['stale'])}</span></div>
    <div class="kv"><span>C&amp;I loans YoY</span><span>{fmt(totci_yoy_val,'%')}{fmt_z(z_last.get('TOTCI_yoy_z'))}{asof_tag(LV['TOTCI']['date'], LV['TOTCI']['stale'])}</span></div>
  </div>
</div>
"""

cols_keys = ["Current","4W","12W","6M","12M","2Y","3Y","4Y"]
display_names = {"Current":"Current","4W":"4W","12W":"12W","6M":"6M",
                 "12M":"12M","2Y":"2Y","3Y":"3Y","4Y":"4Y avg (structural)"}
cols = ["Bucket","Momentum (Now vs "+MOM_COMPARE+")"] + [display_names[k] for k in cols_keys]

html = [styles, '<div class="rot-wrap">', hdr_html, drivers_html,
        '<table class="rot-table"><colgroup>',
        '<col class="col-bucket">','<col class="col-momentum">', *["<col>"]*len(cols_keys),
        '</colgroup><thead><tr>']
for c in cols: html.append(f"<th>{c}</th>")
html.append('</tr></thead><tbody>')

for _, row in table_display.iterrows():
    b = row["Bucket"]
    html.append('<tr>')
    html.append(f'<td class="col-bucket">{b}</td>')
    html.append(f'<td class="col-momentum">{row["Momentum (Now vs "+MOM_COMPARE+")"]}</td>')
    for k in cols_keys:
        val = row[k] if pd.notna(row[k]) else ""
        if val=="":
            html.append("<td></td>")
        else:
            extra = f' {tint_cell(val, table.loc[b, "12M"])}' if k=="Current" else ""
            html.append(f"<td{extra}>{val:.2f}%</td>")
    html.append('</tr>')
html.append('</tbody></table>')

foot = f"""
<div class="foot">
  <b>How to read:</b> Percentages show where market risk appetite sits by bucket across time horizons (left → right).
  <b>Momentum</b> compares now vs {MOM_COMPARE} using a clear pill (Higher/Lower/Near) and the change in percentage points.
  <b>4Y avg (structural)</b> is a long-horizon baseline — where each bucket tends to live across cycles.<br>
  <b>Buckets:</b> {BUCKET_EQ}; {BUCKET_CR}; {BUCKET_RT}; {BUCKET_CMD}; {BUCKET_GLD}; {BUCKET_CRY}; {BUCKET_FX}; {BUCKET_CASH}.<br>
  <b>Regime badge</b> reflects real policy + NFCI stance; driver cards show verdicts, 3-month arrows, and 5-year z-scores with per-series “as of” dates.
</div>
</div>
"""

msg = f"""
🧭 <b>Cycle snapshot – {bucketed.index[-1].date():%b %d %Y}</b><br>
<b>Where we are:</b> Risk-on buckets = {curr_series[['{0}','{1}','{2}','{3}'].format(BUCKET_EQ,BUCKET_CR,BUCKET_CMD,BUCKET_CRY)].sum():.1f}% vs baseline ~{RISK_ON_BASELINE:.0f}%; Risk-off = {curr_series[['{0}','{1}','{2}','{3}'].format(BUCKET_RT,BUCKET_GLD,BUCKET_FX,BUCKET_CASH)].sum():.1f}%.<br>
<b>What’s changing:</b> Top risers vs {MOM_COMPARE} → {', '.join([f'{k} (+{v:.1f}pp)' for k,v in (curr_series-ref_series).sort_values().tail(2).items()])}. 
Top fallers → {', '.join([f'{k} ({v:.1f}pp)' for k,v in (curr_series-ref_series).sort_values().head(2).items()])}.<br>
<b>Why:</b> {', '.join(['real policy restrictive' if rp_v=='tight' else ('financial conditions tight' if nf_v=='tight' else ('curve inverted' if sl_v=='inv' else 'drivers mixed'))])}.
"""

snapshot_html = f"<div class='rot-wrap'><div class='note'>{msg}</div></div>"

rotation_map = pd.DataFrame({
    "Regime":["Tightening","Neutral/Transition","Easing","Reflation","Stress"],
    "Likely next rotation":["Cash → Duration / Credit","Duration → Credit / Equities",
                            "USD → Equities / Commodities","Duration → Commodities / Gold",
                            "Risk → Cash / USD"],
    "Typical outcome":["Carry rally","Valuation expansion","Early-cycle rally",
                       "Inflation hedge bid","Drawdown"]
})
map_html = f"<div class='rot-wrap'><div class='scroll-x'>{rotation_map.to_html(index=False, classes='map-table', escape=False)}</div></div>"

FULL_HTML = "".join(html) + foot + snapshot_html + map_html
with open("report.html", "w", encoding="utf-8") as f:
    f.write(FULL_HTML)

print("✅ wrote report.html")
