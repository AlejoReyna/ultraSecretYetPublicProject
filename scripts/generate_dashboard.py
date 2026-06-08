#!/usr/bin/env python3
"""Generate static HTML trading dashboard from JSONL logs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.settings import load_settings


def _read_jsonl(path: Path, limit: int = 500) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows[-limit:]


def _html_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def render_dashboard(decisions: list[dict], executions: list[dict]) -> str:
    decision_rows = []
    for row in decisions:
        action = str(row.get("action", "WAIT"))
        color = {"ENTER": "#22c55e", "HALT": "#ef4444", "BLOCKED": "#f59e0b"}.get(action, "#94a3b8")
        ml = row.get("ml_ranking") or {}
        ml_score = row.get("ml_confidence")
        if ml_score is None and isinstance(ml, dict):
            scores = ml.get("ml_scores") or {}
            ml_score = max(scores.values()) if scores else None
        decision_rows.append(
            f"<tr style='background:{color}22'>"
            f"<td>{_html_escape(str(row.get('timestamp', '')))}</td>"
            f"<td>{_html_escape(action)}</td>"
            f"<td>{_html_escape(str(row.get('symbol', '-')))}</td>"
            f"<td>{_html_escape(str(row.get('ml_regime', '-')))}</td>"
            f"<td>{ml_score if ml_score is not None else '-'}</td>"
            f"<td>{_html_escape(str(ml.get('ml_selected_symbol', '-')))}</td>"
            f"<td>{_html_escape(str(ml.get('executed_symbol', row.get('symbol', '-'))))}</td>"
            f"<td>{_html_escape(str(row.get('reason', ''))[:80])}</td>"
            "</tr>"
        )

    exec_rows = []
    for row in executions[-50:]:
        tx = row.get("tx_hash", "-")
        exec_rows.append(
            f"<tr><td>{_html_escape(str(row.get('timestamp', '')))}</td>"
            f"<td>{_html_escape(str(row.get('action', '')))}</td>"
            f"<td>{_html_escape(str(row.get('from_symbol', '')))}→{ _html_escape(str(row.get('to_symbol', '')))}</td>"
            f"<td>{_html_escape(str(tx))}</td></tr>"
        )

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Plan B+ Dashboard</title>
<style>
body{{font-family:system-ui,sans-serif;margin:1rem;background:#0f172a;color:#e2e8f0}}
table{{border-collapse:collapse;width:100%;font-size:12px}}
th,td{{border:1px solid #334155;padding:4px 8px;text-align:left}}
h1,h2{{color:#f8fafc}}
</style></head><body>
<h1>Plan B+ Live Dashboard</h1>
<p>Decisions: {len(decisions)} | Executions: {len(executions)}</p>
<h2>Decision timeline (shadow ML scores)</h2>
<table><tr><th>Time</th><th>Action</th><th>Symbol</th><th>ML Regime</th><th>ML Score</th><th>ML Pick</th><th>Executed</th><th>Reason</th></tr>
{''.join(decision_rows) or '<tr><td colspan="8">No decisions</td></tr>'}
</table>
<h2>Recent executions</h2>
<table><tr><th>Time</th><th>Action</th><th>Swap</th><th>Tx</th></tr>
{''.join(exec_rows) or '<tr><td colspan="4">No executions</td></tr>'}
</table>
</body></html>"""


def main() -> int:
    settings = load_settings()
    decisions = _read_jsonl(Path(settings.decision_log_path))
    executions = _read_jsonl(Path(settings.execution_log_path))
    html = render_dashboard(decisions, executions)
    out = Path("dashboard.html")
    out.write_text(html, encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
