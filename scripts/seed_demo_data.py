#!/usr/bin/env python3
"""Seed demo wait sessions into the analytics DB so the reports dashboard
shows realistic-looking data without having to wait for real traffic.

Run on the server:
    sudo -u aesthetics-lab /home/aesthetics-lab/50/.venv/bin/python \
        /home/aesthetics-lab/50/scripts/seed_demo_data.py

Demo rows use label_id >= 90000 so they're easy to remove later:
    sudo sqlite3 /var/lib/ai-track/waittime.db \
        "DELETE FROM wait_sessions WHERE label_id >= 90000;"
"""
from __future__ import annotations

import argparse
import os
import random
import sqlite3
import sys
import time
from datetime import date, datetime, timedelta

# Make `import analytics` work when running this script directly.
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT_ROOT)
import analytics  # noqa: E402


CAMERAS = ["dha-cam1", "dha-cam2", "mm-cam1", "mm-cam2", "fsd1", "fsd2", "fsd3"]

# Per-camera busyness + average-wait profile so the data looks realistic
# (different clinics have different patterns).
PROFILES = {
    "dha-cam1": {"daily_avg_count": 26, "avg_wait_min": 14},
    "dha-cam2": {"daily_avg_count": 30, "avg_wait_min": 11},
    "mm-cam1":  {"daily_avg_count": 22, "avg_wait_min": 9},
    "mm-cam2":  {"daily_avg_count": 28, "avg_wait_min": 12},
    "fsd1":     {"daily_avg_count": 18, "avg_wait_min": 18},
    "fsd2":     {"daily_avg_count": 16, "avg_wait_min": 16},
    "fsd3":     {"daily_avg_count": 14, "avg_wait_min": 22},
}

DEMO_LABEL_OFFSET = 90000          # >= this means demo row, easy to wipe
PER_CAMERA_LABEL_BLOCK = 10000     # each camera gets its own label range


def seed(start: date, end: date, db_path: str, seed_value: int = 42) -> int:
    analytics.init_db(db_path)
    rng = random.Random(seed_value)
    rows = []

    for cam_idx, camera in enumerate(CAMERAS):
        profile = PROFILES.get(camera, {"daily_avg_count": 20, "avg_wait_min": 12})
        avg_count = profile["daily_avg_count"]
        avg_wait_s = profile["avg_wait_min"] * 60

        label_id = DEMO_LABEL_OFFSET + cam_idx * PER_CAMERA_LABEL_BLOCK

        d = start
        while d <= end:
            # Weekend less busy
            day_mult = 0.6 if d.weekday() >= 5 else 1.0
            n_people = max(3, int(rng.gauss(avg_count * day_mult, 4)))

            # Clinic hours 09:00 → 21:00 local time
            day_start_ts = time.mktime(
                datetime(d.year, d.month, d.day, 9, 0).timetuple()
            )
            day_end_ts = time.mktime(
                datetime(d.year, d.month, d.day, 21, 0).timetuple()
            )

            for _ in range(n_people):
                first_seen = rng.uniform(day_start_ts, day_end_ts - 1800)
                wait_s = max(30.0, rng.gauss(avg_wait_s, avg_wait_s * 0.5))
                last_seen = first_seen + wait_s
                rows.append((
                    camera,
                    label_id,
                    first_seen,
                    last_seen,
                    wait_s,
                    d.strftime("%Y-%m-%d"),
                ))
                label_id += 1
            d += timedelta(days=1)

    conn = sqlite3.connect(db_path)
    try:
        conn.executemany(
            "INSERT INTO wait_sessions "
            "(camera, label_id, first_seen, last_seen, total_wait_s, date) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()
    finally:
        conn.close()
    return len(rows)


def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", default="2026-04-11",
                   help="First date (YYYY-MM-DD), default 2026-04-11")
    p.add_argument("--end", default="2026-04-26",
                   help="Last date (YYYY-MM-DD), default 2026-04-26")
    p.add_argument("--db", default=analytics.DEFAULT_DB_PATH,
                   help=f"DB path (default {analytics.DEFAULT_DB_PATH})")
    p.add_argument("--clear", action="store_true",
                   help="Delete existing demo rows (label_id >= 90000) before seeding")
    args = p.parse_args()

    start = parse_date(args.start)
    end = parse_date(args.end)
    if end < start:
        sys.exit("end date must be on or after start date")

    if args.clear:
        analytics.init_db(args.db)
        conn = sqlite3.connect(args.db)
        try:
            cur = conn.execute("DELETE FROM wait_sessions WHERE label_id >= ?",
                               (DEMO_LABEL_OFFSET,))
            conn.commit()
            print(f"Cleared {cur.rowcount} existing demo rows.")
        finally:
            conn.close()

    n = seed(start, end, args.db)
    print(f"Inserted {n} demo sessions across {len(CAMERAS)} cameras "
          f"from {start} to {end}.")
    print(f"Demo data uses label_id >= {DEMO_LABEL_OFFSET}.")
    print("To remove later:")
    print(f"  sudo sqlite3 {args.db} "
          f"\"DELETE FROM wait_sessions WHERE label_id >= {DEMO_LABEL_OFFSET};\"")


if __name__ == "__main__":
    main()
