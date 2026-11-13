"""Resume SSCD encoding from SQLite and publish separate within-shoot groups."""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import msvcrt
import os
from pathlib import Path
import sqlite3
import time
import traceback

import numpy as np
from PIL import Image, ImageOps
import torch
from torchvision import transforms as T

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'sscd_full_run'
from server import DB, folder_key, shoot_key

BATCH = 32
CUTOFF = .80
MODEL = ROOT / 'checkpoint_sscd' / 'sscd_disc_mixup.torchscript.pt'
MODEL_SHA256 = '9f26bd4c848cc19b73d2ae92eea6e04886f61a7b764ceb7a13aeee62e6a6db56'
SCHEMA = """
CREATE TABLE IF NOT EXISTS sscd_embeddings (
    photo_id INTEGER REFERENCES photos(id) ON DELETE CASCADE, encoder TEXT,
    thumbnail_path TEXT NOT NULL, thumbnail_size INTEGER NOT NULL,
    thumbnail_mtime_ns INTEGER NOT NULL, thumbnail_sha256 TEXT NOT NULL,
    vector BLOB NOT NULL CHECK(length(vector)=2048), PRIMARY KEY(photo_id,encoder)
);
CREATE TABLE IF NOT EXISTS sscd_runs (
    run_id TEXT PRIMARY KEY, encoder TEXT NOT NULL, cutoff REAL NOT NULL,
    status TEXT NOT NULL, started_at TEXT NOT NULL, finished_at TEXT,
    total INTEGER NOT NULL, encoded INTEGER NOT NULL DEFAULT 0,
    failed INTEGER NOT NULL DEFAULT 0, group_count INTEGER, collapsed INTEGER
);
CREATE TABLE IF NOT EXISTS sscd_groups (
    run_id TEXT REFERENCES sscd_runs(run_id) ON DELETE CASCADE, group_id INTEGER,
    shoot TEXT NOT NULL, keeper_photo_id INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
    PRIMARY KEY(run_id,group_id)
);
CREATE TABLE IF NOT EXISTS sscd_group_members (
    run_id TEXT, group_id INTEGER, photo_id INTEGER REFERENCES photos(id) ON DELETE CASCADE,
    similarity REAL NOT NULL CHECK(similarity>=0 AND similarity<=1),
    PRIMARY KEY(run_id,photo_id),
    FOREIGN KEY(run_id,group_id) REFERENCES sscd_groups(run_id,group_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS sscd_members_group ON sscd_group_members(run_id,group_id);
"""


def now():
    return datetime.now(timezone.utc).isoformat()


def make_groups(rows, vectors):
    """Assign alternates directly to score-ordered keepers; bound matrix memory."""
    shoots = defaultdict(list)
    for i, row in enumerate(rows):
        shoots[shoot_key(folder_key(row['folder_path']))].append(i)
    for shoot, items in shoots.items():
        def rank(i):
            score = rows[i]['artimuse_score']
            return (-(score if score is not None and math.isfinite(score) else -math.inf), rows[i]['id'])
        order = sorted(items, key=rank)
        features = vectors[order]
        active = np.ones(len(order), dtype=bool)
        for start in range(0,len(order),128):
            if not active[start:start+128].any():
                continue
            sims = features[start:start+128] @ features.T
            for j, values in enumerate(sims,start):
                if not active[j]:
                    continue
                active[j] = False
                matches = np.flatnonzero(active & (values>=CUTOFF))
                active[matches] = False
                if matches.size:
                    yield shoot, rows[order[j]]['id'], [
                        (rows[order[k]]['id'], min(1.,float(values[k]))) for k in matches]


def run():
    if not DB.is_file():
        raise SystemExit('Catalog not found. Run catalog.py or set database in config.json.')
    if not MODEL.is_file():
        raise SystemExit('SSCD checkpoint not found. See the README download instructions.')
    with MODEL.open('rb') as model_file:
        digest = hashlib.file_digest(model_file, 'sha256').hexdigest()
    if digest != MODEL_SHA256:
        raise SystemExit('Unexpected SSCD checkpoint SHA-256. Download the official checkpoint again.')
    encoder = 'sscd-disc-mixup:rgb-exif-skew320-bilinear-imagenet-l2-f32:' + digest
    state = dict(pid=os.getpid(),run_id=now(),encoder=encoder,cutoff=CUTOFF,stage='initializing')
    errors = []
    def status(**changes):
        state.update(changes,updated_at=now())
        temp=OUT/'status.tmp'
        temp.write_text(json.dumps(state,indent=2),encoding='utf-8')
        temp.replace(OUT/'status.json')
        print(json.dumps(state),flush=True)
    with closing(sqlite3.connect(DB,timeout=60)) as db:
        db.row_factory=sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON')
        db.executescript(SCHEMA)
        rows = db.execute("SELECT id,folder_path,thumbnail_path,artimuse_score FROM photos WHERE thumbnail_status='ok' ORDER BY id").fetchall()
        cached = {r['photo_id']:r for r in db.execute('SELECT * FROM sscd_embeddings WHERE encoder=?',(encoder,))}
        total=len(rows)
        if not total:
            raise RuntimeError('No usable thumbnails found')
        with db:
            db.execute("INSERT INTO sscd_runs(run_id,encoder,cutoff,status,started_at,total) VALUES (?,?,?,'running',?,?)",
                       (state['run_id'],encoder,CUTOFF,state['run_id'],total))
        try:
            status(stage='loading_model',total=total,completed=0,failed=0)
            torch.set_num_threads(4)
            model=torch.jit.load(str(MODEL),map_location='cpu').eval().cuda()
            transform=T.Compose([T.Resize((320,320)),T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])
            def load(row):
                try:
                    path=Path(row['thumbnail_path'])
                    st=path.stat()
                    stamp=(str(path),st.st_size,st.st_mtime_ns)
                    data=path.read_bytes()
                    after=path.stat()
                    if (after.st_size,after.st_mtime_ns)!=stamp[1:]:
                        raise ValueError('Thumbnail changed during read')
                    digest=hashlib.sha256(data).hexdigest()
                    previous=cached.get(row['id'])
                    if previous and previous['thumbnail_sha256']==digest:
                        return row,stamp,digest,None,previous['vector'],None
                    with Image.open(io.BytesIO(data)) as im:
                        image=transform(ImageOps.exif_transpose(im).convert('RGB'))
                    return row,stamp,digest,image,None,None
                except (OSError,ValueError,TypeError,Image.DecompressionBombError) as exc:
                    return row,None,None,None,None,str(exc)
            vectors=np.empty((total,512),dtype='<f4')
            valid=np.zeros(total,dtype=bool)
            hits=0
            completed=0
            started=time.perf_counter()
            last=-30
            status(stage='encoding',device=torch.cuda.get_device_name(0))
            with ThreadPoolExecutor(max_workers=4) as pool,torch.inference_mode():
                for i in range(0,total,BATCH):
                    loaded=list(pool.map(load,rows[i:i+BATCH]))
                    new=[record[3] for record in loaded if record[3] is not None]
                    fresh=iter(())
                    if new:
                        features=torch.nn.functional.normalize(model(torch.stack(new).cuda()),dim=1).cpu().numpy()
                        if features.shape!=(len(new),512) or not np.isfinite(features).all():
                            raise RuntimeError('Invalid SSCD output')
                        fresh=iter(features.astype('<f4',copy=False))
                    writes=[]
                    for j,(row,stamp,digest,image,blob,error) in enumerate(loaded,i):
                        if error:
                            errors.append(dict(id=row['id'],error=error))
                            continue
                        if blob is None:
                            vector=next(fresh)
                            blob=vector.tobytes()
                        else:
                            hits+=1
                            vector=np.frombuffer(blob,dtype='<f4')
                        writes.append((row['id'],encoder,*stamp,digest,blob))
                        vectors[j]=vector
                        valid[j]=True
                    completed+=len(writes)
                    with db:
                        db.executemany('INSERT OR REPLACE INTO sscd_embeddings VALUES(?,?,?,?,?,?,?)',writes)
                    elapsed=time.perf_counter()-started
                    if elapsed-last>=15:
                        processed=i+len(loaded)
                        rate=processed/elapsed
                        status(completed=completed,failed=len(errors),cache_hits=hits,photos_per_second=rate,
                               remaining=total-processed,eta_seconds=(total-processed)/rate)
                        last=elapsed
            if errors:
                (OUT/'errors.json').write_text(json.dumps(errors,indent=2),encoding='utf-8')
            if not completed:
                raise RuntimeError('No thumbnails could be encoded')
            status(stage='grouping',completed=completed,failed=len(errors),cache_hits=hits,remaining=0)
            groups=list(make_groups([r for r,ok in zip(rows,valid) if ok],vectors[valid]))
            collapsed=sum(len(matches) for _,_,matches in groups)
            stage='complete_with_errors' if errors else 'complete'
            with db:
                for group_id,(shoot,keeper,members) in enumerate(groups,1):
                    db.execute('INSERT INTO sscd_groups VALUES(?,?,?,?)',(state['run_id'],group_id,shoot,keeper))
                    db.executemany('INSERT INTO sscd_group_members VALUES(?,?,?,?)',
                        [(state['run_id'],group_id,ident,sim) for ident,sim in [(keeper,1.),*members]])
                db.execute("""UPDATE sscd_runs SET status=?,finished_at=?,encoded=?,failed=?,group_count=?,collapsed=?
                           WHERE run_id=?""",(stage,now(),completed,len(errors),len(groups),collapsed,state['run_id']))
            status(stage=stage,groups=len(groups),collapsed=collapsed,eta_seconds=0,finished_at=now())
        except Exception as exc:
            with db:
                db.execute("UPDATE sscd_runs SET status='failed',finished_at=? WHERE run_id=?",(now(),state['run_id']))
            status(stage='failed',error=str(exc),traceback=traceback.format_exc())
            raise


if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    with (OUT/'runner.lock').open('a+b') as lock:
        if lock.tell()==0:
            lock.write(b'0')
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        run()
