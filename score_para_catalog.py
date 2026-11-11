"""Stage full PARA scores, then atomically replace the active ArtiMuse scores."""
import json
import math
import msvcrt
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import time
import traceback

from para_model import load_scorer
from portal_config import catalog_paths

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "para_full_run"
DB,_ = catalog_paths()
STAGING = RUN / "staged_scores.sqlite"
BATCH = 4
INPUT = "1024px catalog thumbnails resized to 448x448 RGB"
ELIGIBLE = "thumbnail_status='ok'"


def now():
    return datetime.now(timezone.utc).isoformat()


def publish(state, staging):
    """Check the target population and replace every eligible score together."""
    with closing(sqlite3.connect(DB, timeout=60)) as db, db:
        db.execute("ATTACH DATABASE ? AS staged", (str(staging),))
        db.execute("BEGIN IMMEDIATE")
        if db.execute("""SELECT 1 FROM artimuse_runs WHERE run_id=? AND status='complete'""",
                      (state["run_id"],)).fetchone():
            return
        eligible = db.execute(f"SELECT COUNT(*) FROM photos WHERE {ELIGIBLE}").fetchone()[0]
        staged = db.execute("SELECT COUNT(*) FROM staged.scores").fetchone()[0]
        matched = db.execute(f"""SELECT COUNT(*) FROM photos p JOIN staged.scores s ON s.id=p.id
            WHERE {ELIGIBLE} AND p.thumbnail_path IS s.thumbnail_path AND s.score BETWEEN 0 AND 100""").fetchone()[0]
        if not eligible or eligible != staged or eligible != matched:
            raise RuntimeError(f"Cannot publish: catalog={eligible}, staged={staged}, valid matches={matched}")
        # Keep a rollback copy in the sidecar, outside the live catalog.
        db.execute("""CREATE TABLE IF NOT EXISTS staged.previous_scores AS
            SELECT p.id, p.artimuse_score, p.artimuse_status, p.artimuse_error, p.artimuse_scored_at
            FROM photos p JOIN staged.scores s ON s.id=p.id""")
        updated = db.execute(f"""UPDATE photos AS p SET artimuse_score=s.score, artimuse_status='ok',
            artimuse_error=NULL, artimuse_scored_at=s.scored_at
            FROM staged.scores s WHERE p.id=s.id AND {ELIGIBLE}""").rowcount
        if updated != eligible:
            raise RuntimeError(f"Unexpected number of updated scores: {updated}")
        db.execute("""INSERT INTO artimuse_runs
            (run_id,started_at,finished_at,status,model,revision,batch_size,preprocessing,precision,eligible,completed,failed,error)
            VALUES (?,?,?,'complete',?,?,?,?,?,?,?,0,NULL)""",
            (state["run_id"],state["started_at"],now(),state["model"],state["revision"],BATCH,INPUT,
             "BF16 weights; FP32 score expectation",eligible,updated))


def main():
    RUN.mkdir(exist_ok=True)
    checkpoint = json.loads((ROOT / "para_revision.json").read_text())
    with (RUN / "runner.lock").open("a+b") as lock:
        lock.seek(0, os.SEEK_END)
        if not lock.tell():
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        state = dict(pid=os.getpid(), started_at=now(), model=checkpoint["repo"], revision=checkpoint["revision"],
                     batch_size=BATCH, database=str(DB), staging_database=str(STAGING), input=INPUT)

        def status(**changes):
            state.update(changes, updated_at=now())
            temp = RUN / "status.tmp"
            temp.write_text(json.dumps(state, indent=2), encoding="utf-8")
            temp.replace(RUN / "status.json")
            print(json.dumps(state), flush=True)

        try:
            with closing(sqlite3.connect(STAGING, timeout=60)) as stage, stage:
                stage.execute("""CREATE TABLE IF NOT EXISTS scores(
                    id INTEGER PRIMARY KEY, thumbnail_path TEXT, score REAL, scored_at TEXT, error TEXT)""")
                stage.execute("CREATE TABLE IF NOT EXISTS manifest(run_id TEXT PRIMARY KEY, revision TEXT NOT NULL)")
                manifest = stage.execute("SELECT run_id,revision FROM manifest").fetchone()
                if manifest:
                    if manifest[1] != checkpoint["revision"]:
                        raise RuntimeError("Staged scores belong to a different checkpoint revision")
                    state["run_id"] = manifest[0]
                else:
                    state["run_id"] = state["started_at"]
                    with closing(sqlite3.connect(DB.as_uri()+"?mode=ro", uri=True, timeout=60)) as source:
                        rows = source.execute(f"SELECT id,thumbnail_path FROM photos WHERE {ELIGIBLE} ORDER BY id").fetchall()
                    if not rows:
                        raise RuntimeError("No catalog photos with usable thumbnails")
                    stage.executemany("INSERT INTO scores(id,thumbnail_path) VALUES (?,?)", rows)
                    stage.execute("INSERT INTO manifest VALUES (?,?)", (state["run_id"],state["revision"]))
                    stage.commit()
                total, completed = stage.execute("SELECT COUNT(*), COUNT(score) FROM scores").fetchone()
                pending = stage.execute("SELECT id,thumbnail_path FROM scores WHERE score IS NULL ORDER BY id").fetchall()
                status(stage="loading_model" if pending else "publishing", total=total, completed=completed,
                       failed=0, percent=100*completed/total, remaining=len(pending), published=False)
                failures = 0
                started = time.perf_counter()
                if pending:
                    torch, pixels, score, model, tokenizer, _, load_time, _ = load_scorer("para")
                    status(stage="scoring", gpu=torch.cuda.get_device_name(0), model_load_seconds=load_time)
                    started = time.perf_counter()
                    last_report = 0

                    def load(item):
                        ident,path = item
                        try:
                            return ident,pixels({"thumbnail_path":path}),None
                        except (OSError, ValueError, TypeError) as exc:
                            return ident,None,str(exc)[:1000]

                    with ThreadPoolExecutor(max_workers=4) as pool:
                        for offset in range(0,len(pending),BATCH):
                            loaded = list(pool.map(load,pending[offset:offset+BATCH]))
                            good = [(i,p) for i,p,error in loaded if error is None]
                            timestamp = now()
                            values = score(torch.stack([p for _,p in good]).pin_memory().to(
                                device="cuda", dtype=torch.bfloat16, non_blocking=True)) if good else []
                            if len(values) != len(good) or not all(math.isfinite(v) and 0 <= v <= 100 for v in values):
                                raise RuntimeError(f"Invalid model scores in batch at {offset}")
                            with stage:
                                stage.executemany("UPDATE scores SET score=?, scored_at=?, error=NULL WHERE id=?",
                                                  [(v,timestamp,i) for (i,_),v in zip(good,values)])
                                stage.executemany("UPDATE scores SET error=? WHERE id=?",
                                                  [(error,i) for i,_,error in loaded if error is not None])
                            completed += len(good)
                            failures += len(loaded)-len(good)
                            elapsed = time.perf_counter()-started
                            if offset == 0 or elapsed-last_report >= 15:
                                attempted = offset+len(loaded)
                                rate = attempted/elapsed
                                status(completed=completed, failed=failures, remaining=len(pending)-attempted,
                                       percent=100*completed/total, photos_per_second=rate,
                                       eta_seconds=(len(pending)-attempted)/rate)
                                last_report = elapsed
                if completed != total:
                    status(stage="incomplete", completed=completed, failed=failures, remaining=total-completed,
                           percent=100*completed/total, published=False)
                    return 1
                status(stage="publishing", completed=completed, failed=0, remaining=0, percent=100.0)
            publish(state, STAGING)
            status(stage="complete", published=True, eta_seconds=0, finished_at=now(),
                   elapsed_scoring_seconds=time.perf_counter()-started)
        except Exception as exc:
            status(stage="failed", error=str(exc), traceback=traceback.format_exc())
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
