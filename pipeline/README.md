# Valet Ops Vision — training pipeline

**Total monthly cost: $0.** No server, no SaaS subscriptions, no API keys you have to pay for. Everything below has a free tier that covers your usage at any reasonable customer count.

## The free stack

| Component | Free service | Why it works for us |
|---|---|---|
| Cron scheduler | **GitHub Actions** | 2,000 min/mo private-repo cap. Daily NHTSA pull ≈ 2 min. |
| State / staging | **JSON files in this repo** | Version-controlled, free, no DB to manage. |
| Photo staging | **GitHub LFS or HF Datasets** | Both free for our volume. |
| Trained model hosting | **GitHub Releases + jsDelivr CDN** | Globally cached, no bandwidth cap for typical traffic. |
| Manifest hosting | **Same repo, served via jsDelivr** | `cdn.jsdelivr.net/gh/<repo>@main/pipeline/manifest.json` |
| Training GPU | **Kaggle** | 30 hrs/week of P100. MobileNetV3 fine-tune ≈ 30 min. |
| Corrections ingest | **Cloudflare Workers + D1** | 100k req/day + 5 GB SQLite, free. |
| LLM auto-filter | **Google Gemini 1.5 Flash** | 1,500 req/day free. |

If your customer count gets very large (10,000+) and Cloudflare D1 or jsDelivr ever hits a paid threshold, the total still tops out around $5-15/mo. The architecture doesn't change — you just upgrade the tier on whichever piece grew.

## Fully automated — no monthly manual step

As of v1.3.1, the entire monthly retrain runs automatically:

- `monthly-train.yml` fires at **12:00 UTC on the 1st of every month**.
- Pushes the Kaggle notebook via the Kaggle API, polls until complete (~30-60 min).
- Pulls the trained ONNX, runs `publish.py` (regression-gated), creates a GitHub Release, bumps `manifest.json`.
- `health-check.yml` runs **daily at 13:00 UTC** — verifies the published manifest is reachable via jsDelivr + the ONNX serves correctly.
- If the health check fails, the workflow **auto-reverts the manifest commit** (rolling customers back to the prior model), opens a GitHub issue, and sends a Slack alert.
- `notify.py` posts success / failure / rollback to a free Slack incoming webhook.

You will not need to touch the pipeline unless you want to. The first time a model lands, you'll get a Slack ping; if anything ever fails, you'll get a Slack ping. Otherwise it just runs.

## Architecture

```
   ┌─────────────────────────────────────────────────────────────────────────┐
   │                                                                         │
   │  GITHUB ACTIONS (free)                                                  │
   │  ─────────────────────                                                  │
   │                                                                         │
   │  daily-pull.yml ─────► nhtsa_pull.py     → pipeline/data/models.json    │
   │                  ─────► wikimedia_fetch.py → pipeline/data/photos/       │
   │                  ─────► llm_filter.py    → verdicts in models.json     │
   │                                          (all changes committed back)  │
   │                                                                         │
   │  publish-model.yml ──► pulls Kaggle output → publish.py → manifest.json │
   │                  ─────► gh release create v{version}                    │
   │                                                                         │
   ├─────────────────────────────────────────────────────────────────────────┤
   │                                                                         │
   │  KAGGLE NOTEBOOK (free GPU)                                             │
   │  ──────────────────────────                                             │
   │  vov-vehicle-classifier — monthly run, exports ONNX to /kaggle/working/ │
   │                                                                         │
   ├─────────────────────────────────────────────────────────────────────────┤
   │                                                                         │
   │  CLOUDFLARE WORKER (free)                                               │
   │  ────────────────────────                                               │
   │  worker/ingest.js  receives POST /v1/submit from customer plugins      │
   │                    stores in Cloudflare D1                              │
   │                                                                         │
   ├─────────────────────────────────────────────────────────────────────────┤
   │                                                                         │
   │  jsDelivr CDN (free)                                                    │
   │  ────────────────────                                                   │
   │  Serves both the manifest and the model ONNX from this repo's            │
   │  GitHub Releases, globally cached, no rate limit for normal traffic.    │
   │                                                                         │
   └─────────────────────────────────────────────────────────────────────────┘

   Customer plugin polls jsDelivr daily → notices new manifest version →
   downloads new ONNX into IndexedDB → swaps in on next scan.
```

## Setup (one-time)

```bash
# 1. Create a GitHub repo for the models (separate from the plugin repo, so
#    customers' plugins don't get bloated by the model history).
#    Name it whatever — e.g. github.com/yourorg/vov-models
#    Copy the contents of this pipeline/ folder into that repo's root.

# 2. Set GitHub secrets in the new repo:
#    Settings → Secrets and variables → Actions → New repository secret
#      GEMINI_API_KEY    (from https://aistudio.google.com/apikey, free)
#      KAGGLE_USERNAME   (from your Kaggle account)
#      KAGGLE_KEY        (from your Kaggle account)
#      SLACK_WEBHOOK     (optional — free Slack incoming webhook for ship + alert notifications)

# 2a. Update pipeline/notebook/kernel-metadata.json with your Kaggle username:
#     Change "REPLACE_USERNAME/vov-vehicle-classifier" → "yourname/vov-vehicle-classifier"

# 3. Set up Cloudflare Worker (5 min):
cd worker
npm install -g wrangler
wrangler login
wrangler d1 create vov-corrections          # creates the D1 DB, prints an ID
# Paste the ID into wrangler.toml -> database_id
wrangler d1 execute vov-corrections --file=./schema.sql
wrangler deploy
# Worker is now live at https://vov-corrections.<your>.workers.dev/v1/submit

# 4. Tell the plugin where to find your manifest + ingest endpoint.
#    In wp-config.php on each customer site (or via mu-plugin):
define( 'VOV_MODEL_REGISTRY',   'https://cdn.jsdelivr.net/gh/yourorg/vov-models@main/pipeline/manifest.json' );
define( 'VOV_CORRECTION_INGEST', 'https://vov-corrections.<your>.workers.dev/v1/submit' );

# 5. Trigger the first NHTSA pull manually to seed the data:
#    Repo → Actions → "Daily — pull NHTSA + fetch new car photos" → Run workflow

# 6. Once you have a few hundred labeled photos, run the Kaggle notebook for
#    the first training. Then trigger "Publish trained model to GitHub Releases".
#    From that moment on, every customer plugin auto-updates within 24 hours.
```

## How new cars get added (concrete walkthrough)

**Scenario: 2027 Hyundai Palisade refresh launches.**

1. **Day 0:** Hyundai registers `2027 Palisade XRT` with NHTSA's VPIC.
2. **Day 1, 04:15 UTC:** `daily-pull.yml` cron fires. `nhtsa_pull.py` sees the new (make, model, year) row, appends it to `pipeline/data/models.json` with `state: pending`. Commit pushed.
3. **Day 1, 04:18 UTC:** Same workflow runs `wikimedia_fetch.py`. Queries Wikimedia Commons for "2027 Hyundai Palisade". Usually 0-3 hits in week 1, 5-15 by week 4.
4. **Day 1, 04:24 UTC:** `llm_filter.py` runs each photo through Gemini Flash. Auto-accepts strong matches, queues the rest as `state: review`.
5. **Meanwhile:** Customer drivers in the wild see new Palisades. AI guesses something close (probably "2024 Palisade"). Manager corrects → plugin POSTs to the Cloudflare Worker → D1 records the correction with anonymized photo URL.
6. **Day 30 (monthly retrain):** You open the Kaggle notebook, click "Run all," wait ~30 min. The notebook pulls accepted Wikimedia photos + verified customer corrections, trains, exports ONNX to `/kaggle/working/`.
7. **Day 30:** You trigger `publish-model.yml` in GitHub with `version: 3.2.0` + changelog. The Action pulls Kaggle output, runs `publish.py` (regression-gated), creates a GitHub Release with the ONNX as an asset, commits the bumped `manifest.json`.
8. **Day 30 + ~10 min:** jsDelivr cache invalidates. The new manifest is live globally.
9. **Day 31:** Every customer plugin's daily registry check sees version 3.2.0, downloads the new ONNX in the background, swaps in on the next scan.

Zero coordination with customers. Zero cost to you. Zero per-scan cost forever.

## Files

| Path | Purpose |
|---|---|
| `nhtsa_pull.py` | Daily NHTSA VPIC pull |
| `wikimedia_fetch.py` | Photo fetch per new model |
| `llm_filter.py` | Gemini auto-verify |
| `publish.py` | Build / bump `manifest.json` |
| `manifest.json` | What customer plugins poll |
| `data/makes.json` | Cumulative NHTSA makes |
| `data/models.json` | Cumulative model-years with state + photo verdicts |
| `data/pull_history.json` | Daily run log |
| `data/photos/` | Staged training photos (commit if small, else mirror to HF) |
| `.github/workflows/daily-pull.yml` | Cron: NHTSA + Wikimedia + Gemini |
| `.github/workflows/publish-model.yml` | Manual trigger after Kaggle run |
| `worker/ingest.js` | Cloudflare Worker — corrections ingest |
| `worker/wrangler.toml` | Worker config |
| `worker/schema.sql` | D1 schema |
| `notebook/README.md` | Kaggle training instructions |
| `MANIFEST_SCHEMA.md` | Wire format the plugin polls |
| `corrections_ingest.py` | (Legacy) Python ingest server — kept as a self-hosted fallback |
| `review_queue.py` | (Legacy) Flask review queue — kept as a self-hosted fallback |
| `train.py` | (Legacy) Standalone training script — superseded by the Kaggle notebook |
