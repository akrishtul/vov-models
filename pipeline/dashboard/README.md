# Status dashboard

Single-file HTML page that shows per-customer health for every site running the Vision plugin. Free to host — published to GitHub Pages by `.github/workflows/deploy-dashboard.yml`.

URL: `https://<your-org>.github.io/<repo-name>/`

## What it shows

- **System overview** — total sites, sites active in last 90 min, current alerts, healthy count, paid-tier count.
- **Per-site row** — traffic-light status (green/yellow/red/grey), site name + URL + tier badge, last heartbeat, scans/hour, success rate, avg latency.
- **Filters** — All / Alerting / Silent / Healthy, plus free-text search by name/URL/license.
- **Detail modal** — drill into any site for plugin/model versions, capture mode + AI slot, license fragment, and the last 48 heartbeats as a sparkline.
- **Auto-refresh every 60 sec** — live indicator in the header.

## Status meaning

| Color | Status | Trigger |
|---|---|---|
| 🟢 Green  | `healthy`  | Latest success rate ≥ 85% (or scan volume < 5/hr — too low to judge) |
| 🟡 Yellow | `warning`  | Success rate 70–85% with ≥ 5 scans/hr |
| 🔴 Red    | `critical` | Success rate < 70% with ≥ 5 scans/hr |
| ⚫ Grey   | `silent`   | No heartbeat in 90+ minutes |

Thresholds are configured in `worker-telemetry/telemetry.js` (`ALERT_THRESHOLDS`).

## First-time setup

When you open the dashboard URL, it prompts for two values (stored in `localStorage`):

1. **Worker API base** — `https://vov-telemetry.<your>.workers.dev`
2. **Dashboard token** — the `DASHBOARD_TOKEN` you set on the telemetry worker with `wrangler secret put DASHBOARD_TOKEN`

After that it's one-click. The token never leaves your browser.

## Slack alerts

The worker's `scheduled` trigger runs anomaly detection every 15 minutes and posts to the `SLACK_WEBHOOK` secret. Alerts are debounced — same site won't re-alert until it self-resolves and re-fails.

Alert shape:
```
🚨 vov status — 1 new alert:
• Capital Grille (VOV-XXXX-XXXX...): CRITICAL — Success rate 42% (< 70%)
   https://capitalgrille.valetops.com
```
