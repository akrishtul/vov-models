/**
 * Corrections ingest — Cloudflare Worker.
 *
 * Receives anonymized correction batches from customer plugins and stores
 * them in a Cloudflare D1 SQLite database. 100,000 requests/day on the
 * Workers free tier, 5 GB on D1 free tier. Easily covers the SaaS for
 * years before you'd need to pay anything.
 *
 * Deploy:
 *   wrangler deploy
 *
 * D1 schema (run once via wrangler d1 execute):
 *   CREATE TABLE corrections (
 *     id INTEGER PRIMARY KEY AUTOINCREMENT,
 *     license TEXT, site TEXT, plugin TEXT,
 *     field TEXT, was TEXT, now_val TEXT,
 *     photo_hash TEXT, photo_url TEXT, region TEXT,
 *     customer_created_at TEXT, received_at TEXT,
 *     verified TEXT DEFAULT 'pending'
 *   );
 *   CREATE INDEX idx_corr_license ON corrections(license);
 *   CREATE INDEX idx_corr_verified ON corrections(verified);
 *
 * Wire wrangler.toml:
 *   name = "vov-corrections"
 *   main = "ingest.js"
 *   compatibility_date = "2026-05-01"
 *   [[d1_databases]]
 *   binding = "DB"
 *   database_name = "vov-corrections"
 *   database_id = "your-d1-id"
 */

export default {
	async fetch(request, env) {
		const url = new URL(request.url);

		if (request.method === "OPTIONS") {
			return new Response(null, { headers: corsHeaders() });
		}
		if (request.method !== "POST") {
			return json({ ok: false, error: "method_not_allowed" }, 405);
		}
		if (url.pathname !== "/v1/submit") {
			return json({ ok: false, error: "not_found" }, 404);
		}

		let body;
		try { body = await request.json(); }
		catch { return json({ ok: false, error: "bad_json" }, 400); }

		const rows = Array.isArray(body.rows) ? body.rows : null;
		if (!rows || rows.length === 0 || rows.length > 100) {
			return json({ ok: false, error: "invalid_rows" }, 400);
		}

		const now = new Date().toISOString();
		const license = String(body.license || "").slice(0, 80);
		const site    = String(body.site    || "").slice(0, 200);
		const plugin  = String(body.plugin  || "").slice(0, 20);

		const stmt = env.DB.prepare(
			`INSERT INTO corrections
			 (license, site, plugin, field, was, now_val,
			  photo_hash, photo_url, region, customer_created_at, received_at, verified)
			 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')`
		);

		const batch = rows.map(r => stmt.bind(
			license,
			site,
			plugin,
			String(r.field   || "").slice(0, 20),
			String(r.was     || "").slice(0, 120),
			String(r.now     || "").slice(0, 120),
			String(r.photo_hash || "").slice(0, 64),
			String(r.photo_url  || "").slice(0, 500),
			String(r.region  || "").slice(0, 12),
			String(r.created_at || "").slice(0, 32),
			now,
		));

		try {
			await env.DB.batch(batch);
		} catch (e) {
			return json({ ok: false, error: "db_error", detail: String(e) }, 500);
		}
		return json({ ok: true, stored: rows.length });
	},
};

function json(body, status = 200) {
	return new Response(JSON.stringify(body), {
		status,
		headers: { "Content-Type": "application/json", ...corsHeaders() },
	});
}
function corsHeaders() {
	return {
		"Access-Control-Allow-Origin":  "*",
		"Access-Control-Allow-Methods": "POST, OPTIONS",
		"Access-Control-Allow-Headers": "Content-Type",
	};
}
