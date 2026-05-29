/**
 * Telemetry ingest + status API Worker.
 *
 * Endpoints:
 *   POST /v1/heartbeat              (no auth)  — customer plugins ping hourly
 *   GET  /api/sites                 (auth)    — list all sites with latest snapshot
 *   GET  /api/sites/:license        (auth)    — per-site detail + last-24h trend
 *   GET  /api/alerts                (auth)    — current alerts
 *   GET  /api/summary               (auth)    — top-line overview for dashboard
 *
 * GET endpoints require Bearer auth via DASHBOARD_TOKEN (set as a Worker secret).
 *
 * Scheduled trigger runs anomaly detection every 15 min; posts to Slack via
 * SLACK_WEBHOOK secret if any sites alert.
 *
 * Deploy:
 *   wrangler deploy
 */

const STATUS = {
	HEALTHY:  "healthy",
	WARNING:  "warning",
	CRITICAL: "critical",
	SILENT:   "silent",
};

const ALERT_THRESHOLDS = {
	silent_minutes:        90,    // no heartbeat in 90 min when site has been active
	success_rate_warn:     0.85,
	success_rate_critical: 0.70,
	low_volume_floor:      5,     // ignore noise — need ≥5 scans/hour to trust the rate
	error_surge_pct:       0.40,  // 40% of scans hitting same error code is alertable
};

export default {
	async fetch(request, env) {
		const url = new URL(request.url);

		if (request.method === "OPTIONS") return cors(new Response(null, { status: 204 }));

		// Public ingest.
		if (request.method === "POST" && url.pathname === "/v1/heartbeat") {
			return cors(await receiveHeartbeat(request, env));
		}

		// Everything else is admin-auth.
		const auth = request.headers.get("authorization") || "";
		const token = auth.replace(/^Bearer\s+/i, "");
		if (!token || token !== env.DASHBOARD_TOKEN) {
			return cors(json({ ok: false, error: "unauthorized" }, 401));
		}

		if (url.pathname === "/api/summary")     return cors(await routeSummary(env));
		if (url.pathname === "/api/sites")       return cors(await routeSites(env));
		if (url.pathname === "/api/alerts")      return cors(await routeAlerts(env));
		const m = url.pathname.match(/^\/api\/sites\/([^/]+)$/);
		if (m) return cors(await routeSiteDetail(env, decodeURIComponent(m[1])));

		return cors(json({ ok: false, error: "not_found" }, 404));
	},

	// Scheduled anomaly check, runs every 15 min (configure in wrangler.toml).
	async scheduled(event, env, ctx) {
		ctx.waitUntil(runAnomalyCheck(env));
	},
};

// ============ Ingest ============
async function receiveHeartbeat(request, env) {
	let body;
	try { body = await request.json(); }
	catch { return json({ ok: false, error: "bad_json" }, 400); }

	let license = String(body.license || "").slice(0, 80); if (!license && body.site && body.site.url) license = String(body.site.url).slice(0, 80);
	const site    = body.site || {};
	if (!license || !site.url) {
		return json({ ok: false, error: "missing_license_or_site" }, 400);
	}
	const now = new Date().toISOString();
	const hour = body.hour || {};
	const model = body.model || {};
	const cum = body.cumulative || {};
	const plugin = body.plugin || {};

	// Compute derived health rating.
	const scans = Number(hour.scans || 0);
	const ok    = Number(hour.success || 0);
	const rate  = scans > 0 ? ok / scans : null;
	const status = computeStatus({ scans, rate, lastSeenMinutesAgo: 0 });

	// Upsert site row + append heartbeat history.
	const stmtSite = env.DB.prepare(`
		INSERT INTO sites (license, url, name, plugin_version, model_version, latest_status,
		                   latest_scans_hour, latest_success_rate, latest_avg_latency_ms,
		                   latest_seen, capture_mode, ai_slot, cloud_optin, license_tier,
		                   first_seen)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
		ON CONFLICT(license) DO UPDATE SET
		  url                 = excluded.url,
		  name                = excluded.name,
		  plugin_version      = excluded.plugin_version,
		  model_version       = excluded.model_version,
		  latest_status       = excluded.latest_status,
		  latest_scans_hour   = excluded.latest_scans_hour,
		  latest_success_rate = excluded.latest_success_rate,
		  latest_avg_latency_ms = excluded.latest_avg_latency_ms,
		  latest_seen         = excluded.latest_seen,
		  capture_mode        = excluded.capture_mode,
		  ai_slot             = excluded.ai_slot,
		  cloud_optin         = excluded.cloud_optin,
		  license_tier        = excluded.license_tier
	`);

	const stmtBeat = env.DB.prepare(`
		INSERT INTO heartbeats (license, received_at, scans, success, fallback_used,
		                        avg_latency_ms, by_provider, by_region, top_errors,
		                        model_version, plugin_version, status)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
	`);

	try {
		await env.DB.batch([
			stmtSite.bind(
				license,
				String(site.url || "").slice(0, 200),
				String(site.name || "").slice(0, 120),
				String(plugin.version || "").slice(0, 20),
				String(model.installed_version || "").slice(0, 20),
				status,
				scans,
				rate,
				Number(hour.avg_latency_ms || 0),
				now,
				String(body.capture_mode || "").slice(0, 20),
				String(body.ai_slot || "").slice(0, 12),
				body.cloud_optin ? 1 : 0,
				String(body.license_tier || "free").slice(0, 12),
				now,    // overwritten on conflict-do-update? No — first_seen is "stuck" unless UPDATE includes it. Good.
			),
			stmtBeat.bind(
				license, now,
				scans, ok,
				Number(hour.fallback_used || 0),
				Number(hour.avg_latency_ms || 0),
				JSON.stringify(hour.by_provider || {}),
				JSON.stringify(hour.by_region || {}),
				JSON.stringify(hour.top_errors || {}),
				String(model.installed_version || ""),
				String(plugin.version || ""),
				status,
			),
		]);
	} catch (e) {
		return json({ ok: false, error: "db_error", detail: String(e) }, 500);
	}

	return json({ ok: true, status });
}

// ============ Read endpoints ============
async function routeSummary(env) {
	const total = (await env.DB.prepare(`SELECT COUNT(*) AS n FROM sites`).first())?.n || 0;
	const recent = (await env.DB.prepare(
		`SELECT COUNT(*) AS n FROM sites WHERE latest_seen > datetime('now','-90 minutes')`
	).first())?.n || 0;
	const status = await env.DB.prepare(
		`SELECT latest_status, COUNT(*) AS n FROM sites GROUP BY latest_status`
	).all();
	const byStatus = {};
	for (const r of status.results || []) byStatus[r.latest_status] = r.n;
	const alerts = (await env.DB.prepare(
		`SELECT COUNT(*) AS n FROM sites WHERE latest_status IN ('warning','critical','silent')`
	).first())?.n || 0;
	const tier = await env.DB.prepare(
		`SELECT license_tier, COUNT(*) AS n FROM sites GROUP BY license_tier`
	).all();
	const byTier = {};
	for (const r of tier.results || []) byTier[r.license_tier || "unknown"] = r.n;

	return json({
		ok:           true,
		total_sites:  total,
		active_sites: recent,
		alerts,
		by_status:    byStatus,
		by_tier:      byTier,
		generated_at: new Date().toISOString(),
	});
}

async function routeSites(env) {
	const rows = await env.DB.prepare(`
		SELECT license, url, name, plugin_version, model_version, latest_status,
		       latest_scans_hour, latest_success_rate, latest_avg_latency_ms,
		       latest_seen, capture_mode, ai_slot, cloud_optin, license_tier
		FROM sites
		ORDER BY
		  CASE latest_status
		    WHEN 'critical' THEN 1
		    WHEN 'silent'   THEN 2
		    WHEN 'warning'  THEN 3
		    WHEN 'healthy'  THEN 4
		    ELSE 5
		  END,
		  latest_seen DESC
	`).all();
	return json({ ok: true, sites: rows.results || [] });
}

async function routeSiteDetail(env, license) {
	const site = await env.DB.prepare(`SELECT * FROM sites WHERE license = ?`).bind(license).first();
	if (!site) return json({ ok: false, error: "not_found" }, 404);

	const trend = await env.DB.prepare(`
		SELECT received_at, scans, success, fallback_used, avg_latency_ms, status, top_errors
		FROM heartbeats
		WHERE license = ?
		ORDER BY received_at DESC
		LIMIT 48
	`).bind(license).all();

	return json({ ok: true, site, trend: trend.results || [] });
}

async function routeAlerts(env) {
	const rows = await env.DB.prepare(`
		SELECT * FROM alerts WHERE resolved_at IS NULL ORDER BY created_at DESC LIMIT 100
	`).all();
	return json({ ok: true, alerts: rows.results || [] });
}

// ============ Anomaly check (scheduled) ============
async function runAnomalyCheck(env) {
	const sites = await env.DB.prepare(`SELECT * FROM sites`).all();
	const now = new Date();
	const fresh = [];

	for (const s of (sites.results || [])) {
		const lastSeen = s.latest_seen ? new Date(s.latest_seen) : null;
		const minutesAgo = lastSeen ? Math.round((now - lastSeen) / 60000) : 999999;

		let status = STATUS.HEALTHY;
		let reason = "";
		if (minutesAgo > ALERT_THRESHOLDS.silent_minutes) {
			status = STATUS.SILENT;
			reason = `No heartbeat in ${minutesAgo} min`;
		} else if (s.latest_scans_hour >= ALERT_THRESHOLDS.low_volume_floor) {
			if (s.latest_success_rate !== null && s.latest_success_rate < ALERT_THRESHOLDS.success_rate_critical) {
				status = STATUS.CRITICAL;
				reason = `Success rate ${(s.latest_success_rate * 100).toFixed(0)}% (< ${ALERT_THRESHOLDS.success_rate_critical * 100}%)`;
			} else if (s.latest_success_rate !== null && s.latest_success_rate < ALERT_THRESHOLDS.success_rate_warn) {
				status = STATUS.WARNING;
				reason = `Success rate ${(s.latest_success_rate * 100).toFixed(0)}% (< ${ALERT_THRESHOLDS.success_rate_warn * 100}%)`;
			}
		}

		// Persist latest_status if changed.
		if (status !== s.latest_status) {
			await env.DB.prepare(`UPDATE sites SET latest_status = ? WHERE license = ?`)
				.bind(status, s.license).run();
		}

		// File alert if not already open + non-healthy.
		if (status !== STATUS.HEALTHY) {
			const open = await env.DB.prepare(
				`SELECT id FROM alerts WHERE license = ? AND status = ? AND resolved_at IS NULL`
			).bind(s.license, status).first();
			if (!open) {
				await env.DB.prepare(`
					INSERT INTO alerts (license, site_name, status, reason, created_at)
					VALUES (?, ?, ?, ?, ?)
				`).bind(s.license, s.name, status, reason, now.toISOString()).run();
				fresh.push({ site: s.name, license: s.license, status, reason, url: s.url });
			}
		} else {
			// Auto-resolve any open alerts for this site.
			await env.DB.prepare(`
				UPDATE alerts SET resolved_at = ?
				WHERE license = ? AND resolved_at IS NULL
			`).bind(now.toISOString(), s.license).run();
		}
	}

	if (fresh.length && env.SLACK_WEBHOOK) {
		await fetch(env.SLACK_WEBHOOK, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({
				text: `:rotating_light: vov status — ${fresh.length} new alert${fresh.length === 1 ? "" : "s"}:\n` +
					fresh.map(a => `• *${a.site}* (${a.license.slice(0, 12)}…): ${a.status.toUpperCase()} — ${a.reason}\n   ${a.url}`).join("\n"),
			}),
		});
	}
}

// ============ helpers ============
function computeStatus({ scans, rate, lastSeenMinutesAgo }) {
	if (lastSeenMinutesAgo > ALERT_THRESHOLDS.silent_minutes) return STATUS.SILENT;
	if (scans < ALERT_THRESHOLDS.low_volume_floor) return STATUS.HEALTHY;
	if (rate < ALERT_THRESHOLDS.success_rate_critical) return STATUS.CRITICAL;
	if (rate < ALERT_THRESHOLDS.success_rate_warn) return STATUS.WARNING;
	return STATUS.HEALTHY;
}

function json(body, status = 200) {
	return new Response(JSON.stringify(body), {
		status,
		headers: { "Content-Type": "application/json" },
	});
}

function cors(resp) {
	const h = new Headers(resp.headers);
	h.set("Access-Control-Allow-Origin",  "*");
	h.set("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
	h.set("Access-Control-Allow-Headers", "Content-Type, Authorization");
	return new Response(resp.body, { status: resp.status, headers: h });
}
