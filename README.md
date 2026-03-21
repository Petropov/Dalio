
# Dalio Rotation Dashboard

Generates `report.html` and emails it weekly each Friday via GitHub Actions.
- Edit schedule in `.github/workflows/dashboard-email.yml`
- Core script: `dashboard.py`

The report opens with a **Decision Strip** that includes the regime call and confidence, top three INCREASE/HOLD/REDUCE actions (sized to sum to roughly zero), and explicit bullets for “what breaks this view?” so portfolio changes are actionable at a glance.

## Weekly PDF

- `rotation_page_weekly.py` renders a print-ready landscape A3 sheet with two portrait A4 weekly cards side-by-side to `output/rotation_weekly.pdf`.
- Run locally: `python rotation_page_weekly.py` after installing dependencies with `pip install -r requirements.txt`.

