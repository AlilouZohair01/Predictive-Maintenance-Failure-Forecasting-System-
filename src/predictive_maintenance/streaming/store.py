"""SQLite persistence with short transactions and fenced, ordered work leases.

The default rollback journal and FULL sync favor a small local installation.
Never hold a transaction open during network requests or model inference.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import time
import uuid

from .alerts import transition


class StreamError(Exception):
    def __init__(self, status, code, message):
        self.status, self.code, self.message = status, code, message


def dumps(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class Store:
    def __init__(self, path, max_pending=5000, max_history=1000, clock=time.time):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_pending, self.max_history, self.clock = max_pending, max_history, clock
        with self.connection() as db:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1):
                raise RuntimeError("Unsupported stream database schema version")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS runs(
                    run_id TEXT PRIMARY KEY, config TEXT NOT NULL, release TEXT NOT NULL,
                    contract TEXT NOT NULL, state TEXT NOT NULL DEFAULT 'open', created_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS units(
                    run_id TEXT NOT NULL REFERENCES runs(run_id), unit_id INTEGER NOT NULL,
                    received_cycle INTEGER NOT NULL DEFAULT 0, processed_cycle INTEGER NOT NULL DEFAULT 0,
                    received_at TEXT NOT NULL, processed_at TEXT, latest TEXT, latest_survival TEXT,
                    PRIMARY KEY(run_id, unit_id));
                CREATE TABLE IF NOT EXISTS readings(
                    run_id TEXT NOT NULL, unit_id INTEGER NOT NULL, cycle INTEGER NOT NULL,
                    body TEXT NOT NULL, received_at TEXT NOT NULL,
                    PRIMARY KEY(run_id,unit_id,cycle),
                    FOREIGN KEY(run_id,unit_id) REFERENCES units(run_id,unit_id));
                CREATE TABLE IF NOT EXISTS jobs(
                    id INTEGER PRIMARY KEY, run_id TEXT NOT NULL, unit_id INTEGER NOT NULL,
                    cycle INTEGER NOT NULL, status TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0,
                    retry_at REAL NOT NULL DEFAULT 0, lease_until REAL NOT NULL DEFAULT 0, owner TEXT,
                    error_code TEXT, UNIQUE(run_id,unit_id,cycle),
                    FOREIGN KEY(run_id,unit_id,cycle) REFERENCES readings(run_id,unit_id,cycle));
                CREATE INDEX IF NOT EXISTS jobs_order ON jobs(run_id,unit_id,cycle,status);
                CREATE INDEX IF NOT EXISTS jobs_ready ON jobs(status,retry_at,lease_until);
                CREATE TABLE IF NOT EXISTS snapshots(
                    run_id TEXT NOT NULL, unit_id INTEGER NOT NULL, cycle INTEGER NOT NULL,
                    body TEXT NOT NULL, processed_at TEXT NOT NULL,
                    PRIMARY KEY(run_id,unit_id,cycle),
                    FOREIGN KEY(run_id,unit_id,cycle) REFERENCES readings(run_id,unit_id,cycle));
                CREATE TABLE IF NOT EXISTS alert_state(
                    run_id TEXT NOT NULL, unit_id INTEGER NOT NULL, rule TEXT NOT NULL, body TEXT NOT NULL,
                    PRIMARY KEY(run_id,unit_id,rule));
                CREATE TABLE IF NOT EXISTS alerts(
                    id INTEGER PRIMARY KEY, run_id TEXT NOT NULL, unit_id INTEGER NOT NULL, rule TEXT NOT NULL,
                    severity TEXT NOT NULL, state TEXT NOT NULL, opened_cycle INTEGER NOT NULL,
                    last_cycle INTEGER NOT NULL, closed_cycle INTEGER, opened_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL, evidence TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS alerts_feed ON alerts(run_id,state,id);
                CREATE TABLE IF NOT EXISTS alert_events(
                    id INTEGER PRIMARY KEY, alert_id INTEGER NOT NULL REFERENCES alerts(id),
                    run_id TEXT NOT NULL, unit_id INTEGER NOT NULL, cycle INTEGER NOT NULL,
                    rule TEXT NOT NULL, transition TEXT NOT NULL, created_at TEXT NOT NULL,
                    UNIQUE(run_id,unit_id,cycle,rule,transition));
                PRAGMA user_version=1;
            """)
            db.execute("INSERT OR IGNORE INTO metadata VALUES('store_id',?)", (str(uuid.uuid4()),))
            self.store_id = db.execute("SELECT value FROM metadata WHERE key='store_id'").fetchone()[0]

    def timestamp(self):
        return datetime.fromtimestamp(self.clock(), timezone.utc).isoformat()

    @contextmanager
    def connection(self, write=False):
        db = sqlite3.connect(self.path, timeout=5, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA synchronous=FULL")
        try:
            if write:
                db.execute("BEGIN IMMEDIATE")
            yield db
            if write:
                db.commit()
        except BaseException:
            if write:
                db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def require_run(db, run_id):
        row = db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            raise StreamError(404, "RUN_NOT_FOUND", "Stream run does not exist.")
        return row

    def create_run(self, config, release, contract):
        body = dumps(config)
        with self.connection(write=True) as db:
            row = db.execute("SELECT * FROM runs WHERE run_id=?", (config["run_id"],)).fetchone()
            if row:
                if row["config"] != body or row["release"] != release:
                    raise StreamError(409, "RUN_CONFLICT", "Run already exists with different settings or model release; use a new run ID.")
            else:
                db.execute("INSERT INTO runs(run_id,config,release,contract,created_at) VALUES(?,?,?,?,?)",
                           (config["run_id"], body, release, dumps(contract), self.timestamp()))
        return self.run(config["run_id"])

    def run(self, run_id):
        with self.connection() as db:
            row = self.require_run(db, run_id)
            counts = dict(db.execute("SELECT status,COUNT(*) FROM jobs WHERE run_id=? GROUP BY status", (run_id,)).fetchall())
            units = db.execute("SELECT COUNT(*) FROM units WHERE run_id=?", (run_id,)).fetchone()[0]
        progress = "blocked" if counts.get("failed") else "draining" if row["state"] == "closed" and counts.get("pending", 0) + counts.get("processing", 0) else "completed" if row["state"] == "closed" else "accepting"
        return {"run_id": run_id, "config": json.loads(row["config"]), "release_version": row["release"],
                "contract": json.loads(row["contract"]), "state": row["state"], "progress": progress,
                "created_at": row["created_at"], "unit_count": units, "jobs": {k: counts.get(k, 0) for k in ("pending", "processing", "done", "failed")}}

    def runs(self, limit=100, after=""):
        with self.connection() as db:
            rows = db.execute("SELECT run_id FROM runs WHERE run_id>? ORDER BY run_id LIMIT ?", (after, limit)).fetchall()
        return [self.run(r[0]) for r in rows]

    def close_run(self, run_id):
        with self.connection(write=True) as db:
            self.require_run(db, run_id)
            db.execute("UPDATE runs SET state='closed' WHERE run_id=?", (run_id,))
        return self.run(run_id)

    def ingest(self, run_id, message):
        unit, cycle = message["unit_id"], message["reading"]["cycle"]
        body, received = dumps(message["reading"]), self.timestamp()
        with self.connection(write=True) as db:
            run = self.require_run(db, run_id)
            old = db.execute("SELECT body FROM readings WHERE run_id=? AND unit_id=? AND cycle=?", (run_id, unit, cycle)).fetchone()
            if old:
                if old[0] != body:
                    raise StreamError(409, "READING_CONFLICT", "A different reading is already stored for this engine/cycle.")
                status = db.execute("SELECT status FROM jobs WHERE run_id=? AND unit_id=? AND cycle=?", (run_id, unit, cycle)).fetchone()[0]
                return {"run_id": run_id, "unit_id": unit, "cycle": cycle, "disposition": "duplicate", "job_status": status}
            if run["state"] != "open":
                raise StreamError(409, "RUN_CLOSED", "Run is closed to new readings; create a new run for another replay.")
            limit = min(self.max_history, json.loads(run["contract"])["max_history_cycles"])
            if cycle > limit:
                raise StreamError(422, "HISTORY_LIMIT", "Engine history exceeds the current anomaly model serving limit.")
            previous = db.execute("SELECT received_cycle FROM units WHERE run_id=? AND unit_id=?", (run_id, unit)).fetchone()
            expected = previous[0] + 1 if previous else 1
            if cycle != expected:
                raise StreamError(409, "CYCLE_OUT_OF_ORDER", f"Next required cycle for this engine is {expected}.")
            queued = db.execute("SELECT COUNT(*) FROM jobs WHERE status!='done'").fetchone()[0]
            if queued >= self.max_pending:
                raise StreamError(429, "STREAM_BACKLOG_FULL", "Processing backlog is full; retry this reading later.")
            if previous is None:
                count = db.execute("SELECT COUNT(*) FROM units WHERE run_id=?", (run_id,)).fetchone()[0]
                if count >= 1000:
                    raise StreamError(422, "UNIT_LIMIT", "A run supports at most 1000 engines.")
                db.execute("INSERT INTO units(run_id,unit_id,received_at) VALUES(?,?,?)", (run_id, unit, received))
            db.execute("INSERT INTO readings VALUES(?,?,?,?,?)", (run_id, unit, cycle, body, received))
            db.execute("INSERT INTO jobs(run_id,unit_id,cycle) VALUES(?,?,?)", (run_id, unit, cycle))
            db.execute("UPDATE units SET received_cycle=?,received_at=? WHERE run_id=? AND unit_id=?", (cycle, received, run_id, unit))
        return {"run_id": run_id, "unit_id": unit, "cycle": cycle, "disposition": "accepted", "job_status": "pending"}

    def claim(self, run_id=None, lease_seconds=180):
        now = self.clock()
        with self.connection(write=True) as db:
            db.execute("INSERT INTO metadata VALUES('worker_last_claim_at',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (self.timestamp(),))
            row = db.execute("""SELECT j.* FROM jobs j WHERE
                ((j.status='pending' AND j.retry_at<=?) OR (j.status='processing' AND j.lease_until<=?))
                AND (? IS NULL OR j.run_id=?)
                AND NOT EXISTS(SELECT 1 FROM jobs p WHERE p.run_id=j.run_id AND p.unit_id=j.unit_id AND p.cycle<j.cycle AND p.status!='done')
                ORDER BY j.id LIMIT 1""", (now, now, run_id, run_id)).fetchone()
            if row is None:
                return None
            owner = str(uuid.uuid4())
            db.execute("UPDATE jobs SET status='processing',owner=?,lease_until=?,attempts=attempts+1 WHERE id=?", (owner, now + lease_seconds, row["id"]))
            return {**dict(row), "owner": owner, "attempts": row["attempts"] + 1}

    def status(self):
        with self.connection() as db:
            counts = dict(db.execute("SELECT status,COUNT(*) FROM jobs GROUP BY status").fetchall())
            alerts = dict(db.execute("SELECT state,COUNT(*) FROM alerts GROUP BY state").fetchall())
            heartbeat = db.execute("SELECT value FROM metadata WHERE key='worker_last_claim_at'").fetchone()
        return {"store_id": self.store_id, "schema_version": 1,
                "jobs": {state: counts.get(state, 0) for state in ("pending", "processing", "done", "failed")},
                "alerts": {state: alerts.get(state, 0) for state in ("active", "resolved")},
                "worker_last_claim_at": heartbeat[0] if heartbeat else None}

    @staticmethod
    def owned(db, job):
        row = db.execute("SELECT * FROM jobs WHERE id=? AND owner=? AND status='processing'", (job["id"], job["owner"])).fetchone()
        if row is None:
            raise StreamError(409, "LEASE_LOST", "Another worker owns this job; its result was not committed.")
        return row

    def renew(self, job, lease_seconds=180):
        with self.connection(write=True) as db:
            self.owned(db, job)
            db.execute("UPDATE jobs SET lease_until=? WHERE id=?", (self.clock() + lease_seconds, job["id"]))

    def history(self, run_id, unit, through=1000, after=0, limit=1000):
        with self.connection() as db:
            self.require_unit(db, run_id, unit)
            rows = db.execute("SELECT body,received_at FROM readings WHERE run_id=? AND unit_id=? AND cycle>? AND cycle<=? ORDER BY cycle LIMIT ?",
                              (run_id, unit, after, through, limit)).fetchall()
        return [{"reading": json.loads(r[0]), "received_at": r[1]} for r in rows]

    def fail(self, job, code, retryable, max_attempts=5):
        with self.connection(write=True) as db:
            current = self.owned(db, job)
            retry = retryable and current["attempts"] < max_attempts
            db.execute("UPDATE jobs SET status=?,retry_at=?,owner=NULL,lease_until=0,error_code=? WHERE id=?",
                       ("pending" if retry else "failed", self.clock() + min(30, 2 ** min(current["attempts"], 5)), code, job["id"]))

    def retry_failed(self, run_id, unit):
        with self.connection(write=True) as db:
            self.require_unit(db, run_id, unit)
            count = db.execute("UPDATE jobs SET status='pending',attempts=0,retry_at=0,error_code=NULL WHERE run_id=? AND unit_id=? AND status='failed'", (run_id, unit)).rowcount
        return {"run_id": run_id, "unit_id": unit, "retried_jobs": count}

    def complete(self, job, snapshot):
        now = self.timestamp()
        run_id, unit, cycle = job["run_id"], job["unit_id"], job["cycle"]
        with self.connection(write=True) as db:
            self.owned(db, job)
            run = self.require_run(db, run_id)
            if snapshot["release_version"] != run["release"] or snapshot["cycle"] != cycle or snapshot["unit_id"] != unit:
                raise ValueError("Result does not match its pinned job")
            body = dumps(snapshot)
            db.execute("INSERT INTO snapshots VALUES(?,?,?,?,?)", (run_id, unit, cycle, body, now))
            db.execute("UPDATE units SET processed_cycle=?,processed_at=?,latest=? WHERE run_id=? AND unit_id=?", (cycle, now, body, run_id, unit))
            if snapshot.get("survival") is not None:
                db.execute("UPDATE units SET latest_survival=? WHERE run_id=? AND unit_id=?", (dumps(snapshot["survival"]), run_id, unit))
            policy = json.loads(run["config"])["alert_policy"]
            rul = snapshot["prediction"]["rul_cycles"] if snapshot.get("prediction") else None
            anomaly = snapshot["anomaly"]["latest"] if snapshot.get("anomaly") else None
            for rule in ("low_rul", "sustained_anomaly"):
                previous = db.execute("SELECT body FROM alert_state WHERE run_id=? AND unit_id=? AND rule=?", (run_id, unit, rule)).fetchone()
                state, event = transition(rule, json.loads(previous[0]) if previous else None, cycle, rul, anomaly, policy)
                if event == "raised":
                    evidence = dumps({"rul_cycles": rul, "anomaly": anomaly, "policy": policy, "release_version": run["release"]})
                    cursor = db.execute("INSERT INTO alerts(run_id,unit_id,rule,severity,state,opened_cycle,last_cycle,opened_at,updated_at,evidence) VALUES(?,?,?,?,?,?,?,?,?,?)",
                                        (run_id, unit, rule, "critical" if rule == "low_rul" else "warning", "active", cycle, cycle, now, now, evidence))
                    state["alert_id"] = cursor.lastrowid
                if state["active"]:
                    db.execute("UPDATE alerts SET last_cycle=?,updated_at=? WHERE id=?", (cycle, now, state["alert_id"]))
                if event == "resolved":
                    db.execute("UPDATE alerts SET state='resolved',closed_cycle=?,last_cycle=?,updated_at=? WHERE id=?", (cycle, cycle, now, state["alert_id"]))
                if event:
                    db.execute("INSERT INTO alert_events(alert_id,run_id,unit_id,cycle,rule,transition,created_at) VALUES(?,?,?,?,?,?,?)", (state["alert_id"], run_id, unit, cycle, rule, event, now))
                db.execute("INSERT INTO alert_state VALUES(?,?,?,?) ON CONFLICT(run_id,unit_id,rule) DO UPDATE SET body=excluded.body", (run_id, unit, rule, dumps(state)))
            db.execute("UPDATE jobs SET status='done',owner=NULL,lease_until=0,error_code=NULL WHERE id=?", (job["id"],))

    @staticmethod
    def require_unit(db, run_id, unit):
        row = db.execute("SELECT * FROM units WHERE run_id=? AND unit_id=?", (run_id, unit)).fetchone()
        if row is None:
            raise StreamError(404, "UNIT_NOT_FOUND", "Engine does not exist in this stream run.")
        return row

    def latest(self, run_id, unit, stale_seconds=30):
        with self.connection() as db:
            run = self.require_run(db, run_id)
            row = self.require_unit(db, run_id, unit)
            blocked = db.execute("SELECT cycle,status,error_code,attempts FROM jobs WHERE run_id=? AND unit_id=? AND status!='done' ORDER BY cycle LIMIT 1", (run_id, unit)).fetchone()
            active = db.execute("SELECT id,rule,severity,opened_cycle FROM alerts WHERE run_id=? AND unit_id=? AND state='active' ORDER BY id", (run_id, unit)).fetchall()
        age = max(0, self.clock() - datetime.fromisoformat(row["received_at"]).timestamp())
        return {"run_id": run_id, "unit_id": unit, "dataset": "FD001", "release_version": run["release"],
                "received_cycle": row["received_cycle"], "processed_cycle": row["processed_cycle"],
                "processing_lag_cycles": row["received_cycle"] - row["processed_cycle"],
                "last_received_at": row["received_at"], "last_processed_at": row["processed_at"],
                "seconds_since_receipt": age, "is_stale": run["state"] == "open" and age > stale_seconds,
                "source_state": run["state"], "next_job": dict(blocked) if blocked else None,
                "latest": json.loads(row["latest"]) if row["latest"] else None,
                "latest_survival": json.loads(row["latest_survival"]) if row["latest_survival"] else None,
                "active_alerts": [dict(r) for r in active]}

    def units(self, run_id, limit=100, after=0, stale_seconds=30):
        with self.connection() as db:
            self.require_run(db, run_id)
            ids = db.execute("SELECT unit_id FROM units WHERE run_id=? AND unit_id>? ORDER BY unit_id LIMIT ?", (run_id, after, limit)).fetchall()
        return [self.latest(run_id, r[0], stale_seconds) for r in ids]

    def predictions(self, run_id, unit, after=0, limit=100):
        with self.connection() as db:
            self.require_unit(db, run_id, unit)
            rows = db.execute("SELECT body,processed_at FROM snapshots WHERE run_id=? AND unit_id=? AND cycle>? ORDER BY cycle LIMIT ?", (run_id, unit, after, limit)).fetchall()
        return [{**json.loads(r[0]), "processed_at": r[1]} for r in rows]

    def alerts(self, run_id, state="active", after=0, limit=100):
        with self.connection() as db:
            self.require_run(db, run_id)
            rows = db.execute("SELECT * FROM alerts WHERE run_id=? AND (?='all' OR state=?) AND id>? ORDER BY id LIMIT ?", (run_id, state, state, after, limit)).fetchall()
        return [{**dict(r), "evidence": json.loads(r["evidence"])} for r in rows]

    def events(self, run_id, after=0, limit=100):
        with self.connection() as db:
            self.require_run(db, run_id)
            return [dict(r) for r in db.execute("SELECT * FROM alert_events WHERE run_id=? AND id>? ORDER BY id LIMIT ?", (run_id, after, limit))]
