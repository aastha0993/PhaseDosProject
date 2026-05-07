#!/usr/bin/env python3
"""
app.py  —  Neobrutalist UI server for the Synthetic Cancel Culture Sandbox.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

load_dotenv()

app = Flask(__name__)
_program = None  # lazy-loaded DSPy module


def _get_program():
    global _program
    if _program is None:
        import dspy
        from compiler import (
            OPTIMIZED_STATE_PATH,
            STUDENT_MODEL,
            PRDisasterAnalyzer,
            build_lm,
        )
        dspy.configure(lm=build_lm(STUDENT_MODEL))
        _program = PRDisasterAnalyzer()
        if OPTIMIZED_STATE_PATH.exists():
            _program.load(str(OPTIMIZED_STATE_PATH))
    return _program


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/creators")
def list_creators():
    db = Path("sandbox_data/creators.db")
    if not db.exists():
        return jsonify([])
    with sqlite3.connect(db) as conn:
        rows = conn.execute(
            "SELECT name, niche, drama_score, bot_percentage, avg_sponsorship_roi, "
            "follower_count, apology_video_count FROM creators ORDER BY drama_score DESC"
        ).fetchall()
    return jsonify([
        {
            "name": r[0], "niche": r[1], "drama_score": r[2],
            "bot_pct": r[3], "roi": r[4], "followers": r[5], "apologies": r[6],
        }
        for r in rows
    ])


@app.route("/api/analyze", methods=["POST"])
def analyze():
    body = request.get_json(force=True, silent=True) or {}
    name = (body.get("creator_name") or "").strip()
    if not name:
        return jsonify({"error": "creator_name is required"}), 400
    try:
        from compiler import analyze_creator
        result = analyze_creator(name, program=_get_program())
        result["metric_passed"] = bool(result.get("metric_passed"))
        result["cancel_velocity"] = float(result.get("cancel_velocity") or 0.0)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5050))
    print(f"\n  CANCEL CULTURE SANDBOX  ▶  http://localhost:{port}\n")
    app.run(debug=False, port=port)
