"""Reusable PDF drawing primitives for rotation reports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from reportlab.lib import colors
from reportlab.pdfgen.canvas import Canvas


@dataclass(frozen=True)
class Theme:
    background: colors.Color = colors.HexColor("#0b0d11")
    card: colors.Color = colors.HexColor("#111827")
    ink: colors.Color = colors.HexColor("#e5e7eb")
    muted: colors.Color = colors.HexColor("#9ca3af")
    accent: colors.Color = colors.HexColor("#4cc9f0")
    ok: colors.Color = colors.HexColor("#10b981")
    warn: colors.Color = colors.HexColor("#f59e0b")
    danger: colors.Color = colors.HexColor("#ef4444")


THEME = Theme()


def draw_card(
    c: Canvas,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    value: str,
    subtitle: str | None = None,
    bar_value: float | None = None,
    icon: str | None = None,
    theme: Theme = THEME,
) -> None:
    """Draw a dashboard-style card with optional bar and icon."""

    c.saveState()
    c.setFillColor(theme.card)
    c.roundRect(x, y - h, w, h, 10, fill=True, stroke=False)

    padding = 10
    cursor_y = y - padding

    c.setFillColor(theme.muted)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(x + padding, cursor_y, title.upper())

    cursor_y -= 20
    c.setFillColor(theme.ink)
    c.setFont("Helvetica-Bold", 22)
    c.drawString(x + padding, cursor_y, value)

    if icon:
        c.setFont("Helvetica-Bold", 16)
        c.drawString(x + w - padding - c.stringWidth(icon, "Helvetica-Bold", 16), cursor_y, icon)

    if subtitle:
        cursor_y -= 14
        c.setFillColor(theme.muted)
        c.setFont("Helvetica", 9)
        c.drawString(x + padding, cursor_y, subtitle)

    if bar_value is not None:
        clamped = max(0.0, min(1.0, bar_value))
        bar_w = w - 2 * padding
        bar_h = 8
        bar_y = y - h + padding + bar_h
        c.setFillColor(theme.background)
        c.roundRect(x + padding, bar_y, bar_w, bar_h, 4, fill=True, stroke=False)
        c.setFillColor(theme.accent)
        c.roundRect(x + padding, bar_y, bar_w * clamped, bar_h, 4, fill=True, stroke=False)

    c.restoreState()


def draw_action_row(
    c: Canvas,
    x: float,
    y: float,
    width: float,
    height: float,
    label: str,
    delta_text: str,
    delta_value: float,
    cap: float,
    crowded: bool = False,
    theme: Theme = THEME,
) -> None:
    """Render a single action row with an inline bar."""

    c.saveState()
    c.setFillColor(theme.card)
    c.roundRect(x, y - height, width, height, 6, fill=True, stroke=False)

    padding = 8
    text_y = y - padding - 2
    c.setFillColor(theme.ink)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(x + padding, text_y, label)

    delta_x = x + width - padding - c.stringWidth(delta_text, "Helvetica-Bold", 16)
    c.setFont("Helvetica-Bold", 16)
    c.setFillColor(theme.ok if delta_value > 0 else (theme.danger if delta_value < 0 else theme.muted))
    c.drawString(delta_x, text_y, delta_text)

    bar_width = width - 2 * padding
    bar_height = 6
    bar_y = y - height + padding + bar_height
    c.setFillColor(theme.background)
    c.roundRect(x + padding, bar_y, bar_width, bar_height, 3, fill=True, stroke=False)

    if cap > 0:
        frac = min(1.0, abs(delta_value) / cap)
        fill_w = bar_width * frac
        bar_color = theme.ok if delta_value > 0 else theme.danger
        if delta_value == 0:
            bar_color = theme.muted
        c.setFillColor(bar_color)
        c.roundRect(x + padding, bar_y, fill_w, bar_height, 3, fill=True, stroke=False)

    if crowded:
        badge = "Crowded ⚠"
        c.setFont("Helvetica-Bold", 8)
        badge_w = c.stringWidth(badge, "Helvetica-Bold", 8) + 10
        badge_h = 14
        badge_x = x + padding
        badge_y = y - height + padding + badge_h + 2
        c.setFillColor(theme.warn)
        c.roundRect(badge_x, badge_y - badge_h, badge_w, badge_h, 4, fill=True, stroke=False)
        c.setFillColor(theme.background)
        c.drawString(badge_x + 5, badge_y - 4, badge)

    c.restoreState()


def draw_signal_matrix(
    c: Canvas,
    x: float,
    y: float,
    width: float,
    row_height: float,
    rows: Iterable[dict],
    theme: Theme = THEME,
) -> None:
    """Render a compact alignment matrix with arrows and a single numeric column."""

    c.saveState()
    padding = 6
    columns = [
        ("Bucket", 0.38),
        ("Curr %", 0.14),
        ("ST", 0.12),
        ("MT", 0.12),
        ("LT", 0.12),
        ("Align", 0.12),
    ]

    header_y = y
    c.setFont("Helvetica-Bold", 9)
    c.setFillColor(theme.muted)
    offset = x + padding
    for title, frac in columns:
        c.drawString(offset, header_y, title)
        offset += frac * width

    current_y = y - row_height
    for idx, row in enumerate(rows):
        c.setFillColor(colors.Color(0, 0, 0, alpha=0.0))
        if idx % 2 == 0:
            c.setFillColor(colors.Color(1, 1, 1, alpha=0.05))
            c.roundRect(x, current_y - row_height + padding / 2, width, row_height - padding, 4, fill=True, stroke=False)

        c.setFillColor(theme.ink)
        c.setFont("Helvetica", 10)

        col_offsets: list[float] = []
        running = x + padding
        for _, frac in columns:
            col_offsets.append(running)
            running += frac * width

        c.drawString(col_offsets[0], current_y + row_height / 2, row.get("bucket", ""))
        c.setFillColor(theme.muted)
        c.drawRightString(col_offsets[1] + columns[1][1] * width - 6, current_y + row_height / 2, row.get("current", ""))
        c.setFillColor(theme.accent)
        c.drawString(col_offsets[2], current_y + row_height / 2, row.get("st", ""))
        c.drawString(col_offsets[3], current_y + row_height / 2, row.get("mt", ""))
        c.drawString(col_offsets[4], current_y + row_height / 2, row.get("lt", ""))
        c.setFillColor(theme.ink)
        c.drawRightString(col_offsets[5] + columns[5][1] * width - 6, current_y + row_height / 2, row.get("align", ""))

        if row.get("crowded"):
            badge = f"⚠ {row.get('crowd_z', '')}"
            c.setFont("Helvetica-Bold", 7)
            badge_w = c.stringWidth(badge, "Helvetica-Bold", 7) + 8
            c.setFillColor(theme.warn)
            c.roundRect(x + width - badge_w - padding, current_y + row_height / 2 - 6, badge_w, 12, 3, fill=True, stroke=False)
            c.setFillColor(theme.background)
            c.drawString(x + width - badge_w - padding + 4, current_y + row_height / 2 - 2, badge)

        current_y -= row_height

    c.restoreState()


def draw_callout_box(
    c: Canvas,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    bullets: list[str],
    theme: Theme = THEME,
) -> None:
    """Render a small callout with bullet points."""

    c.saveState()
    c.setFillColor(theme.card)
    c.roundRect(x, y - h, w, h, 8, fill=True, stroke=False)

    padding = 10
    cursor_y = y - padding
    c.setFillColor(theme.warn)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(x + padding, cursor_y, title)

    c.setFillColor(theme.ink)
    c.setFont("Helvetica", 9)
    for bullet in bullets[:2]:
        cursor_y -= 14
        c.drawString(x + padding + 6, cursor_y, f"• {bullet}")

    c.restoreState()
