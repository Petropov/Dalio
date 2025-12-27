import math
from pathlib import Path

import pytest

pytest.importorskip("reportlab")

from rotation_decision_engine import SignalBundle, arrow_for
from rotation_page_weekly import build_action_rows, render_weekly_pdf, sanitize_delta, sanitize_number


def _bundle(name: str, align: float) -> SignalBundle:
    return SignalBundle(
        bucket=name,
        st=align,
        mt=align,
        lt=align,
        align=align,
        align_arrow=arrow_for(align),
        crowd_z=None,
        crowded=False,
        effective_score=align,
        current_weight=10.0,
    )


def test_sanitize_numbers_never_emit_nan():
    assert sanitize_number(float("nan")) == "—"
    assert sanitize_number(float("inf")) == "—"
    assert sanitize_delta(float("nan")) == "—"
    assert "—" not in sanitize_number(1.234)


def test_action_ordering_respects_effective_score():
    bundles = [
        _bundle("A", 0.6),
        _bundle("B", 0.4),
        _bundle("C", -0.2),
        _bundle("D", -0.8),
        _bundle("E", 0.05),
    ]
    rows = build_action_rows(bundles)
    assert rows[0]["bucket"] == "A"
    assert rows[-1]["bucket"] == "D"


def test_render_handles_missing_history(tmp_path: Path):
    bundles = [_bundle("A", 0.3), _bundle("B", -0.1), _bundle("C", 0.0)]
    output = tmp_path / "weekly.pdf"
    render_weekly_pdf(
        bundles,
        rotation_level=0.0,
        rotation_wow=0.0,
        risk_on_share=50.0,
        regime="Defensive tilt",
        confidence=5.0,
        breakers=["Test one", "Test two"],
        output_path=output,
    )
    assert output.exists()
    assert output.stat().st_size > 0
