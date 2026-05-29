"""
Tiny Flask app — your weekly 5-10 min thumbs-up/down on photos the LLM
filter wasn't confident about.

Run: pip install flask && python review_queue.py
Then visit http://your-vps:8001/

UI: photo + LLM verdict + 2 buttons (accept / reject). Hotkey 1 = accept,
2 = reject.
"""

import os
import sqlite3
from datetime import datetime, timezone

from flask import Flask, Response, redirect, request, send_file

DB_PATH = os.environ.get("VOV_PIPELINE_DB", "/var/lib/vov/pipeline.sqlite3")
app = Flask(__name__)


def get_conn():
    return sqlite3.connect(DB_PATH)


@app.route("/")
def home():
    conn = get_conn()
    row = conn.execute("""
        SELECT pr.photo_id, pr.photo_path, pr.llm_confidence, pr.llm_reason,
               makes.make, models.model, models.year
        FROM photo_review pr
        JOIN models ON models.model_id = pr.model_id
        JOIN makes  ON makes.make_id   = models.make_id
        WHERE pr.llm_verdict = 'review' AND pr.human_verdict IS NULL
        ORDER BY pr.photo_id ASC
        LIMIT 1
    """).fetchone()
    pending = conn.execute(
        "SELECT COUNT(*) FROM photo_review WHERE llm_verdict='review' AND human_verdict IS NULL"
    ).fetchone()[0]

    if not row:
        return Response("<h2>Queue empty 🎉</h2>", mimetype="text/html")

    pid, path, conf, reason, make, model, year = row
    html = f"""
    <html><head><title>VOV review queue</title>
    <style>
      body {{ font-family: -apple-system, sans-serif; background: #1e2532; color: #fff; margin: 0; padding: 24px; }}
      .card {{ max-width: 720px; margin: 0 auto; background: #2a3450; border-radius: 16px; padding: 24px; }}
      img {{ width: 100%; border-radius: 12px; }}
      .meta {{ color: #aab; font-size: 13.5px; margin-top: 12px; }}
      .label {{ font-size: 24px; font-weight: 700; margin-top: 14px; }}
      .row {{ display: flex; gap: 10px; margin-top: 18px; }}
      button {{ flex: 1; padding: 16px; border: 0; border-radius: 10px; font-weight: 700; font-size: 16px; cursor: pointer; }}
      .accept {{ background: #15a692; color: #fff; }}
      .reject {{ background: #c62828; color: #fff; }}
      .pending {{ position: fixed; top: 16px; right: 24px; background: #e69b1e; color: #1f1408; padding: 6px 12px; border-radius: 99px; font-weight: 700; font-size: 12px; }}
    </style></head><body>
    <div class="pending">{pending} pending</div>
    <div class="card">
      <img src="/photo/{pid}" />
      <div class="label">{make} {model} ({year})</div>
      <div class="meta">LLM confidence: {conf:.2f} · {reason}</div>
      <form class="row" method="post" action="/verdict">
        <input type="hidden" name="photo_id" value="{pid}" />
        <button type="submit" name="v" value="reject" class="reject">2 — Reject</button>
        <button type="submit" name="v" value="accept" class="accept">1 — Accept</button>
      </form>
    </div>
    <script>
    document.addEventListener('keydown', e => {{
      if (e.key === '1') document.querySelector('.accept').click();
      if (e.key === '2') document.querySelector('.reject').click();
    }});
    </script>
    </body></html>
    """
    return Response(html, mimetype="text/html")


@app.route("/photo/<int:pid>")
def photo(pid: int):
    conn = get_conn()
    row = conn.execute("SELECT photo_path FROM photo_review WHERE photo_id = ?", (pid,)).fetchone()
    if not row:
        return Response(status=404)
    return send_file(row[0], mimetype="image/jpeg")


@app.route("/verdict", methods=["POST"])
def verdict():
    pid = int(request.form["photo_id"])
    v   = request.form["v"]
    if v not in ("accept", "reject"):
        return Response(status=400)
    conn = get_conn()
    conn.execute(
        "UPDATE photo_review SET human_verdict=?, human_reviewed_at=? WHERE photo_id=?",
        (v, datetime.now(timezone.utc).isoformat(), pid),
    )
    conn.commit()
    return redirect("/")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8001)))
