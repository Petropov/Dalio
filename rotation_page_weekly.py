"""PDF renderer for the Cross-Asset Rotation weekly page."""

from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import yfinance as yf
from reportlab.lib.pagesizes import landscape, letter
from reportlab.pdfgen import canvas

from pdf_components import THEME, draw_action_row, draw_callout_box, draw_card, draw_signal_matrix
from rotation_decision_engine import (
    BUCKETS,
    CYCLICAL,
    DEFENSIVE,
    HORIZONS,
    RISK_ON,
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


DATA_DIR = Path("data")
OUT_DIR = Path("output")
STATE_DIR = Path("state")

for directory in (DATA_DIR, OUT_DIR, STATE_DIR):
    directory.mkdir(exist_ok=True)

SNAPSHOT_PATH = STATE_DIR / "last_snapshot.json"
HISTORY_PATH = DATA_DIR / "risk_on_share.csv"


def sanitize_number(value: float | None, suffix: str = "") -> str:
    if value is None or not math.isfinite(value):
        return "—"
    if suffix:
        return f"{value:.2f}{suffix}"
    return f"{value:.2f}"


def sanitize_delta(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "—"
    return f"{value:+.2f} pp"


def _softmax_weights(momentums: list[float]) -> list[float]:
    scores = [max(0.0, 0.0 if math.isnan(v) else v) for v in momentums]
    if not scores:
        return []
    if all(abs(s) < 1e-6 for s in scores):
        return [1.0 / len(scores)] * len(scores)
    temperature = pd.Series(scores).std() + 1e-6
    exp_scores = [math.exp(s / temperature) for s in scores]
    total = sum(exp_scores)
    return [s / total for s in exp_scores]


def _ret(series: pd.Series, window: int) -> float:
    series = series.dropna()
    if len(series) <= window:
        return float("nan")
    latest = series.iloc[-1]
    prev = series.iloc[-1 - window]
    try:
        return float((latest / prev - 1.0) * 100.0)
    except Exception:
        return float("nan")


def _nanmean(values: Iterable[float]) -> float:
    filtered = [v for v in values if not pd.isna(v)]
    if not filtered:
        return float("nan")
    return float(sum(filtered) / len(filtered))


def _last_close(series: pd.Series) -> float:
    series = series.dropna()
    return float(series.iloc[-1]) if not series.empty else float("nan")


def _ret_at_position(series: pd.Series, position: int, window: int) -> float:
    if position <= window:
        return float("nan")
    latest = series.iloc[position]
    prev = series.iloc[position - window]
    try:
        return float((latest / prev - 1.0) * 100.0)
    except Exception:
        return float("nan")


def compute_weight_history(prices: pd.DataFrame) -> dict[str, list[float]]:
    if prices.empty:
        return {bucket: [] for bucket, _ in BUCKETS}
    weekly = prices.resample("W-FRI").last().ffill().dropna(how="all")
    weekly = weekly.tail(200)
    weekly_windows = {label: max(1, window // 5) for label, window in HORIZONS.items()}

    history_weights: dict[str, list[float]] = {bucket: [] for bucket, _ in BUCKETS}
    for pos in range(len(weekly)):
        momentums: list[float] = []
        for bucket, _ in BUCKETS:
            series = weekly[bucket].fillna(method="ffill")
            returns = {label: _ret_at_position(series, pos, weekly_windows[label]) for label in HORIZONS}
            short = _nanmean([returns["4W"], returns["12W"]])
            medium = _nanmean([returns["6M"], returns["12M"]])
            long_term = _nanmean([returns["2Y"], returns["3Y"]])
            long_penalty = 0.0 if pd.isna(long_term) else max(0.0, -long_term)
            momentum = _nanmean([short, medium, medium])
            if not pd.isna(momentum):
                momentum -= 0.25 * long_penalty
            momentums.append(momentum if not pd.isna(momentum) else 0.0)

        weights_row = _softmax_weights(momentums)
        for idx, (bucket, _) in enumerate(BUCKETS):
            if len(weights_row) > idx:
                history_weights[bucket].append(round(float(weights_row[idx] * 100.0), 4))
    return history_weights


def compute_snapshot() -> tuple[pd.DataFrame, dict[str, list[float]], float, float, float]:
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

    rows: list[dict[str, float | str]] = []
    momentum_vector: list[float] = []
    for bucket, _ in BUCKETS:
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
                **{key: (None if pd.isna(value) else round(value, 2)) for key, value in returns.items()},
                "CurrentClose": None if pd.isna(current) else round(current, 2),
            }
        )

    df = pd.DataFrame(rows)
    weights = _softmax_weights(momentum_vector)
    df["Current"] = [round(w * 100.0, 2) for w in weights]
    df = df.drop(columns=["CurrentClose"])

    # WoW deltas
    prior: dict[str, float] = {}
    if SNAPSHOT_PATH.exists():
        try:
            prior = json.loads(SNAPSHOT_PATH.read_text())
        except Exception:
            prior = {}

    def _pp_delta(curr: float, prev: float | None) -> float:
        try:
            return round(float(curr) - float(prev), 2)
        except Exception:
            return float("nan")

    df["WoW"] = [
        _pp_delta(current, prior.get(bucket) if isinstance(prior, dict) else None)
        for bucket, current in zip(df["Bucket"], df["Current"], strict=True)
    ]

    rotation_level = round(df.loc[df["Bucket"].isin(DEFENSIVE), "Current"].fillna(0).sum() - df.loc[df["Bucket"].isin(CYCLICAL), "Current"].fillna(0).sum(), 2)
    rotation_wow = round(df.loc[df["Bucket"].isin(DEFENSIVE), "WoW"].fillna(0).sum() - df.loc[df["Bucket"].isin(CYCLICAL), "WoW"].fillna(0).sum(), 2)
    risk_on_share = round(df.loc[df["Bucket"].isin(RISK_ON), "Current"].fillna(0).sum(), 2)

    # persist snapshot & history
    SNAPSHOT_PATH.write_text(json.dumps({row.Bucket: float(row.Current) for _, row in df.iterrows()}))
    history_row = pd.DataFrame([
        {"date": dt.date.today().isoformat(), "risk_on_pct": risk_on_share, "ri_level": rotation_level, "ri_wow": rotation_wow}
    ])
    if HISTORY_PATH.exists():
        history = pd.read_csv(HISTORY_PATH)
        history = pd.concat([history, history_row], ignore_index=True)
        history = history.sort_values("date").drop_duplicates("date", keep="last")
    else:
        history = history_row
    history.to_csv(HISTORY_PATH, index=False)

    return df, compute_weight_history(price_df), rotation_level, rotation_wow, risk_on_share


def prepare_bundles(df: pd.DataFrame, weight_history: dict[str, list[float]]) -> List[SignalBundle]:
    bundles: list[SignalBundle] = []
    for _, record in df.iterrows():
        momentums = {key: record.get(key) for key in HORIZONS}
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
    return bundles


def build_action_rows(bundles: List[SignalBundle]) -> list[dict]:
    plan = recommendation_plan(bundles)
    bundle_map = {b.bucket: b for b in bundles}
    rows: list[dict] = []
    for category in ("INCREASE", "HOLD", "REDUCE"):
        for bucket, delta in plan[category]:
            bundle = bundle_map[bucket]
            rows.append(
                {
                    "category": category,
                    "bucket": bucket,
                    "delta": delta,
                    "delta_text": sanitize_delta(delta),
                    "crowded": bundle.crowded,
                    "crowd_z": None if bundle.crowd_z is None else round(bundle.crowd_z, 2),
                }
            )
    return rows


def build_matrix_rows(bundles: List[SignalBundle]) -> list[dict]:
    rows: list[dict] = []
    for bundle in bundles:
        rows.append(
            {
                "bucket": bundle.bucket,
                "current": sanitize_number(bundle.current_weight, suffix="%"),
                "st": arrow_for(bundle.st),
                "mt": arrow_for(bundle.mt),
                "lt": arrow_for(bundle.lt),
                "align": sanitize_number(bundle.align),
                "crowded": bundle.crowded,
                "crowd_z": "" if bundle.crowd_z is None else f"{bundle.crowd_z:.2f}",
            }
        )
    return rows


def render_weekly_pdf(
    bundles: List[SignalBundle],
    rotation_level: float,
    rotation_wow: float,
    risk_on_share: float,
    regime: str,
    confidence: float,
    breakers: list[str],
    output_path: Path,
) -> None:
    page_width, page_height = landscape(letter)
    margin = 36
    c = canvas.Canvas(str(output_path), pagesize=(page_width, page_height))
    c.setFillColor(THEME.background)
    c.rect(0, 0, page_width, page_height, stroke=False, fill=True)

    title_y = page_height - margin
    c.setFont("Helvetica-Bold", 22)
    c.setFillColor(THEME.ink)
    c.drawString(margin, title_y, "Cross-Asset Capital Rotation — Weekly")
    c.setFont("Helvetica", 10)
    c.setFillColor(THEME.muted)
    c.drawString(margin, title_y - 16, f"As of {dt.date.today().isoformat()}")

    card_width = (page_width - margin * 2 - 16 * 2) / 3
    card_height = 110
    top_y = title_y - 30
    risk_state = "Low" if risk_on_share < 40 else ("Medium" if risk_on_share < 60 else "High")
    risk_color = "🟢" if risk_state == "Low" else ("🟠" if risk_state == "Medium" else "🔴")
    draw_card(
        c,
        margin,
        top_y,
        card_width,
        card_height,
        "Regime",
        regime,
        f"Confidence {sanitize_number(confidence)}/10",
        bar_value=None if not math.isfinite(confidence) else confidence / 10,
    )
    delta_icon = "↑" if rotation_wow > 0 else ("↓" if rotation_wow < 0 else "→")
    draw_card(
        c,
        margin + card_width + 16,
        top_y,
        card_width,
        card_height,
        "Rotation index",
        f"{sanitize_delta(rotation_level).replace(' pp', ' pp')}",
        f"1W: {sanitize_delta(rotation_wow)}",
        icon=delta_icon,
    )
    draw_card(
        c,
        margin + (card_width + 16) * 2,
        top_y,
        card_width,
        card_height,
        "Risk-on share",
        f"{sanitize_number(risk_on_share, suffix='%')}",
        f"Bias {risk_state}",
        icon=risk_color,
    )

    # Actions section
    action_y = top_y - card_height - 24
    c.setFillColor(THEME.ink)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, action_y, "THIS WEEK — DO THIS")
    rows = build_action_rows(bundles)
    inc_rows = [row for row in rows if row["category"] == "INCREASE"]
    hold_rows = [row for row in rows if row["category"] == "HOLD"]
    reduce_rows = [row for row in rows if row["category"] == "REDUCE"]
    filtered_rows = inc_rows[:3] + hold_rows[:2] + reduce_rows[:3]
    action_height = 38
    start_y = action_y - 18
    for idx, row in enumerate(filtered_rows):
        if idx >= 8:
            break
        y_pos = start_y - idx * (action_height + 6)
        draw_action_row(
            c,
            margin,
            y_pos,
            page_width - 2 * margin,
            action_height,
            row["bucket"],
            row["delta_text"],
            0.0 if not math.isfinite(row["delta"]) else float(row["delta"]),
            cap=1.5,
            crowded=row["crowded"],
        )

    # Bottom matrix and callout
    matrix_y = start_y - len(filtered_rows) * (action_height + 6) - 12
    c.setFont("Helvetica-Bold", 12)
    c.setFillColor(THEME.ink)
    c.drawString(margin, matrix_y, "Why we tilt this way")
    matrix_rows = build_matrix_rows(bundles)
    draw_signal_matrix(c, margin, matrix_y - 14, page_width - 2 * margin - 180, 22, matrix_rows)

    callout_x = page_width - margin - 170
    callout_y = matrix_y
    draw_callout_box(c, callout_x, callout_y, 170, 80, "What breaks this view?", breakers)

    c.showPage()
    c.save()


def main() -> None:  # pragma: no cover - manual entry point
    df, weight_history, rotation_level, rotation_wow, risk_on_share = compute_snapshot()
    bundles = prepare_bundles(df, weight_history)
    regime, confidence = regime_call(bundles, CYCLICAL, DEFENSIVE)
    breakers = what_breaks_view(regime)
    output_path = OUT_DIR / "rotation_weekly.pdf"
    render_weekly_pdf(bundles, rotation_level, rotation_wow, risk_on_share, regime, confidence, breakers, output_path)
    print(f"✅ wrote {output_path}")


if __name__ == "__main__":  # pragma: no cover - manual entry point
    main()
