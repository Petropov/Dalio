
# Dalio Rotation Dashboard

Generates `report.html` and emails it weekly each Friday via GitHub Actions.
- Edit schedule in `.github/workflows/dashboard-email.yml`
- Core script: `dashboard.py`

The report opens with a **Decision Strip** that includes the regime call and confidence, top three INCREASE/HOLD/REDUCE actions (sized to sum to roughly zero), and explicit bullets for “what breaks this view?” so portfolio changes are actionable at a glance.

