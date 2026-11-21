"""Rename the local school demo's master data, keeping event history immutable.

Dry-run by default. --apply requires an external backup directory. This script
does not ingest events, start analyses, approve actions or rewrite snapshots.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
from datetime import datetime, timezone

REGION = "MANUAL_SCHOOL_WALKTHROUGH_20261111"
ROAD = "MANUAL_SCHOOL_ROAD_20261111"
POI = "MANUAL_SCHOOL_POI_20261111"
RENAMES = [
    ("regions", "region_id", REGION, "手动流程验证区（虚构）", "校园周边片区"),
    ("roads", "road_id", ROAD, "手动验证学校门口路", "学校门前路"),
    ("pois", "poi_id", POI, "手动验证学校（虚构）", "校园小学"),
]


def protected_digest(conn):
    excluded = {"regions", "roads", "pois", "road_aliases"}
    result = {}
    for (table,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"):
        if table in excluded:
            continue
        quoted = '"' + table.replace('"', '""') + '"'
        rows = sorted(json.dumps(list(row), ensure_ascii=False, default=lambda v: v.hex(),
                                 separators=(",", ":")) for row in conn.execute(f"SELECT * FROM {quoted}"))
        result[table] = {"count": len(rows), "sha256": hashlib.sha256("\n".join(rows).encode()).hexdigest()}
    return result


def normalize(db: Path, *, apply=False, backup_dir: Path | None = None):
    if not db.is_file():
        raise ValueError("Database must already exist")
    conn = sqlite3.connect(db.resolve().as_uri() + ("?mode=rw" if apply else "?mode=ro"), uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    backup = None
    try:
        if apply:
            if backup_dir is None:
                raise ValueError("--backup-dir is required with --apply")
            backup_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            backup = backup_dir / f"before_location_names_{stamp}.db"
            if backup.exists():
                raise ValueError("Refusing to overwrite a backup")
            with sqlite3.connect(backup) as destination:
                conn.backup(destination)
            conn.execute("BEGIN IMMEDIATE")
        else:
            conn.execute("BEGIN")
        active = conn.execute("SELECT count(*) FROM collaboration_runs WHERE status IN ('running','pending','routing','arbitrating','fusing')").fetchone()[0]
        active += conn.execute("SELECT count(*) FROM workflow_runs WHERE status='running'").fetchone()[0]
        if active:
            raise ValueError("Wait for active analysis/workflow runs before normalizing master data")
        before = protected_digest(conn)
        changes = []
        for table, key, identity, old, new in RENAMES:
            row = conn.execute(f"SELECT * FROM {table} WHERE {key}=?", (identity,)).fetchone()
            if row is None or row["name"] not in (old, new):
                raise ValueError(f"Unexpected {table} record; refusing to guess")
            if row["verification_status"] != "synthetic":
                raise ValueError("Only the known synthetic location may be renamed")
            if table != "regions" and row["region_id"] != REGION:
                raise ValueError("Unexpected region ownership")
            changes.append({"table": table, "id": identity, "before": row["name"], "after": new})
        collisions = conn.execute("SELECT road_id FROM road_aliases WHERE region_id=? AND normalized_alias=? AND road_id<>? AND status='active'",
                                  (REGION, "学校门前路", ROAD)).fetchall()
        if collisions:
            raise ValueError("New road name conflicts with another active road alias")
        legacy_alias = conn.execute("SELECT 1 FROM road_aliases WHERE road_id=? AND alias=? AND status='active'",
                                    (ROAD, RENAMES[1][3])).fetchone()
        if not legacy_alias:
            raise ValueError("Legacy road alias must remain available for historical events")
        if apply:
            now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            for table, key, identity, old, new in RENAMES:
                conn.execute(f"UPDATE {table} SET name=?,updated_at=? WHERE {key}=? AND name<>?", (new, now, identity, new))
            conn.execute("UPDATE regions SET city=?,description=? WHERE region_id=?",
                         ("示例城市", "校园周边交通事件演示区域；位置资料尚未经过实际地理核验。", REGION))
            conn.execute("INSERT INTO road_aliases (alias_id,region_id,road_id,alias,normalized_alias,source_type,status,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT(region_id,road_id,alias) DO NOTHING",
                         ("roadalias_school_display_20261111", REGION, ROAD, "学校门前路", "学校门前路", "manual_synthetic_validation", "active", now, now))
            after = protected_digest(conn)
            if before != after:
                raise ValueError("Protected event/history tables changed; rolling back")
            if conn.execute("PRAGMA foreign_key_check").fetchall():
                raise ValueError("Foreign key check failed")
            conn.commit()
        else:
            conn.rollback()
        return {"status": "APPLIED" if apply else "DRY_RUN", "database": str(db.resolve()),
                "backup": str(backup) if backup else None, "changes": changes,
                "protectedTablesUnchanged": True if apply else None, "protectedTables": before,
                "legacyAliasPreserved": True, "analysisStarted": False}
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--backup-dir", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = normalize(args.db, apply=args.apply, backup_dir=args.backup_dir)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "protectedTables"}, ensure_ascii=False, indent=2))
