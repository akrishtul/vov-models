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

		// v2026.05.30 — Training-side export endpoint. Returns verified corrections
		// after the given `?after=<id>` cursor, up to `?limit` rows (default 500,
		// max 1000). Bearer-auth via env.EXPORT_TOKEN (set via `wrangler secret put
		// EXPORT_TOKEN`).
		if (request.method === "GET" && url.pathname === "/v1/export") {
			const auth = request.headers.get("Authorization") || "";
			const expected = env.EXPORT_TOKEN ? `Bearer ${env.EXPORT_TOKEN}` : null;
			if (!expected || auth !== expected) {
				return json({ ok: false, error: "unauthorized" }, 401);
			}
			const after = parseInt(url.searchParams.get("after") || "0", 10) || 0;
			const limit = Math.max(1, Math.min(1000, parseInt(url.searchParams.get("limit") || "500", 10) || 500));
			try {
				const { results } = await env.DB.prepare(
					`SELECT id, license, site, plugin,
					        field, was, now_val,
					        photo_hash, photo_url, region,
					        customer_created_at, received_at, verified
					   FROM corrections
					  WHERE id > ?
					    AND verified = 'verified'
					    AND photo_url <> ''
					    AND photo_hash <> ''
					  ORDER BY id ASC
					  LIMIT ?`
				).bind(after, limit).all();
				// Surface the field/was/now triplet in a shape the training pipeline likes:
				// it cares about (now_make, now_model). Field shape from VOV plugin is
				// `field='make_model', now='Toyota|Camry'` (pipe-separated).
				const rows = (results || []).map(r => {
					let now_make = "", now_model = "";
					if (r.field === "make_model" && typeof r.now_val === "string" && r.now_val.includes("|")) {
						const [mk, md] = r.now_val.split("|", 2);
						now_make = (mk || "").trim();
						now_model = (md || "").trim();
					}
					return {
						id: r.id,
						license: r.license,
						site: r.site,
						plugin: r.plugin,
						field: r.field,
						was: r.was,
						now: r.now_val,
						now_make,
						now_model,
						photo_hash: r.photo_hash,
						photo_url: r.photo_url,
						region: r.region,
						customer_created_at: r.customer_created_at,
						received_at: r.received_at,
					};
				});
				return json({ ok: true, count: rows.length, rows });
			} catch (e) {
				return json({ ok: false, error: "db_error", detail: String(e) }, 500);
			}
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
