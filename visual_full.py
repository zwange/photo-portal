"""Resumable CPU-only visual grouping. Uses the pilot's hashing and verification."""
import argparse,collections,concurrent.futures,datetime,functools,itertools,json,msvcrt,os,sqlite3,time,uuid
from pathlib import Path
from visual_features import VERSION,hashing,visual,verify
from portal_config import catalog_paths

ROOT=Path(__file__).resolve().parent
DB,_=catalog_paths()
from portal_config import load_config,is_visible

OUT=ROOT/'visual_full_run'
MASKS=tuple(sum(1<<bit for bit in positions) for count in range(4) for positions in itertools.combinations(range(16),count))

def indexed_candidates(items):
    """Exact recall for 64-bit Hamming distance <=14 via four 16-bit indexes.

    At least one block has distance <=3; otherwise total distance is >=16.
    Only candidates from those buckets need a full-hash distance calculation.
    """
    buckets=[collections.defaultdict(list) for _ in range(4)]
    prior={}
    for ident,value in items:
        candidates=set()
        for part in range(4):
            block=(value>>(part*16))&65535;index=buckets[part]
            for mask in MASKS:
                found=index.get(block^mask)
                if found:candidates.update(found)
        matches=[(other,(value^prior[other]).bit_count()) for other in candidates if (value^prior[other]).bit_count()<=14]
        yield ident,matches
        prior[ident]=value
        for part in range(4):buckets[part][(value>>(part*16))&65535].append(ident)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-id');ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    OUT.mkdir(exist_ok=True)
    lock=(OUT/'run.lock').open('a+b');lock.seek(0)
    if not lock.read(1):lock.write(b'0');lock.flush()
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    run=args.run_id or (json.loads((OUT/'launch.json').read_text())['run_id'] if args.resume else datetime.datetime.now(datetime.timezone.utc).strftime('full-%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:6])
    folder=OUT/run;folder.mkdir(exist_ok=True)
    if not args.resume:(folder/'stop.request').unlink(missing_ok=True)
    started=time.monotonic();stage='starting';details={};errors=0
    def status(force=False):
        nonlocal last_status
        if not force and time.monotonic()-last_status<5:return
        value=dict(run_id=run,pid=os.getpid(),stage=stage,threads=4,updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_seconds=round(time.monotonic()-started,2),errors=errors,**details)
        temp=OUT/'status.tmp';temp.write_text(json.dumps(value,indent=2));os.replace(temp,OUT/'status.json')
        print(json.dumps(value),flush=True);last_status=time.monotonic()
    def stop():return (folder/'stop.request').exists()
    last_status=0
    db=sqlite3.connect(DB,timeout=120);db.row_factory=sqlite3.Row;db.execute('PRAGMA foreign_keys=ON')
    db.executescript('''CREATE TABLE IF NOT EXISTS visual_hash_errors(run_id TEXT,photo_id INTEGER,error TEXT,PRIMARY KEY(run_id,photo_id));
    CREATE INDEX IF NOT EXISTS idx_visual_match_work ON visual_matches(run_id,decision,photo_a,photo_b);''')
    manifest=folder/'manifest.json'
    if manifest.exists():data=json.loads(manifest.read_text());rows=data['photos']
    else:
        hidden=load_config()['hidden_folders']
        rows=[dict(r) for r in db.execute("SELECT id,file_name,full_path,folder_path,thumbnail_path,artimuse_score,width_px,height_px FROM photos WHERE thumbnail_status='ok' AND artimuse_status='ok' ORDER BY id") if is_visible(r['folder_path'],hidden)]
        data=dict(run_id=run,scope='full',photos=rows,hidden_folders=hidden,hash_policy=VERSION,verification_policy='pilot-visual-v1',candidate_distance=14,threads=4)
        manifest.write_text(json.dumps(data,indent=2),encoding='utf-8')
    byid={r['id']:r for r in rows}
    with db:db.execute('INSERT OR IGNORE INTO visual_hash_runs VALUES(?,?,?,?)',(run,datetime.datetime.now(datetime.timezone.utc).isoformat(),'full',json.dumps(dict(state='running',manifest=str(manifest)))))
    (OUT/'launch.json').write_text(json.dumps(dict(pid=os.getpid(),run_id=run,folder=str(folder),threads=4,script=str(Path(__file__))),indent=2))
    try:
        stage='hashing';hashes={};todo=[]
        cached={r['photo_id']:dict(r) for r in db.execute('SELECT * FROM photo_visual_hashes WHERE version=?',(VERSION,))}
        for row in rows:
            try:
                st=Path(row['thumbnail_path']).stat();h=cached.get(row['id'])
                if h and h['thumbnail_size']==st.st_size and h['thumbnail_mtime_ns']==st.st_mtime_ns:hashes[row['id']]=h
                else:todo.append(row)
            except OSError as error:
                with db:db.execute('INSERT OR REPLACE INTO visual_hash_errors VALUES(?,?,?)',(run,row['id'],str(error)))
                errors+=1
        details=dict(total_photos=len(rows),hashed=len(hashes),hashes_to_compute=len(todo));status(True)
        def hash_safe(row):
            try:return hashing(row),None
            except Exception as error:return dict(photo_id=row['id']),str(error)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            iterator=iter(todo);pending={pool.submit(hash_safe,r) for r in itertools.islice(iterator,8)}
            batch=[]
            while pending:
                done,pending=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    h,error=future.result();ident=h['photo_id']
                    if error:
                        with db:db.execute('INSERT OR REPLACE INTO visual_hash_errors VALUES(?,?,?)',(run,ident,error))
                        errors+=1
                    else:hashes[ident]=h;batch.append((ident,VERSION,h['phash64'],h['phash256'],h['dhash64'],h['decoded_sha256'],h['thumbnail_size'],h['thumbnail_mtime_ns']))
                    if not stop():
                        row=next(iterator,None)
                        if row:pending.add(pool.submit(hash_safe,row))
                if len(batch)>=64 or not pending:
                    with db:db.executemany('INSERT OR REPLACE INTO photo_visual_hashes VALUES(?,?,?,?,?,?,?,?)',batch)
                    batch=[]
                details['hashed']=len(hashes);status()
        if stop():stage='stopped';status(True);return
        stage='candidate_search';details.update(indexed=0,candidate_pairs=0);status(True)
        batch=[]
        items=((ident,int(hashes[ident]['phash64'],16)) for ident in sorted(hashes))
        for number,(ident,matches) in enumerate(indexed_candidates(items),1):
            for other,d64 in matches:
                a,b=sorted((ident,other));ha,hb=hashes[a],hashes[b]
                metrics=dict(a=a,b=b,phash64_distance=d64,phash256_distance=(int(ha['phash256'],16)^int(hb['phash256'],16)).bit_count(),dhash64_distance=(int(ha['dhash64'],16)^int(hb['dhash64'],16)).bit_count())
                batch.append((run,a,b,'pending',json.dumps(metrics)))
            if len(batch)>=1000 or number==len(hashes):
                with db:db.executemany('INSERT OR IGNORE INTO visual_matches VALUES(?,?,?,?,?)',batch)
                batch=[]
            details.update(indexed=number,candidate_pairs=details['candidate_pairs']+len(matches));status()
            if stop():stage='stopped';status(True);return
        stage='verification'
        counts=dict(db.execute('SELECT decision,COUNT(*) FROM visual_matches WHERE run_id=? GROUP BY decision',(run,)))
        total=sum(counts.values());completed=total-counts.get('pending',0)
        details.update(candidate_pairs=total,verified_processed=completed,verified_accepted=counts.get('verified_auto',0));status(True)
        @functools.lru_cache(maxsize=384)
        def feature(ident):return visual(byid[ident])
        class Features:
            def __getitem__(self,ident):return feature(ident)
        features=Features()
        def verify_safe(r):
            m=json.loads(r['metrics_json']);a,b=r['photo_a'],r['photo_b']
            try:return verify((a,b,m['phash64_distance'],m['phash256_distance'],m['dhash64_distance']),features,hashes)
            except Exception as error:return dict(m,decision='error',reason=str(error))
        while not stop():
            work=[dict(r) for r in db.execute("SELECT photo_a,photo_b,metrics_json FROM visual_matches WHERE run_id=? AND decision='pending' ORDER BY photo_a,photo_b LIMIT 256",(run,))]
            if not work:break
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                iterator=iter(work);pending={pool.submit(verify_safe,r) for r in itertools.islice(iterator,8)}
                batch=[]
                while pending:
                    done,pending=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
                    for future in done:
                        m=future.result();batch.append((m['decision'],json.dumps(m),run,m['a'],m['b']))
                        completed+=1;details['verified_accepted']+=m['decision']=='verified_auto';errors+=m['decision']=='error'
                        if not stop():
                            item=next(iterator,None)
                            if item:pending.add(pool.submit(verify_safe,item))
                    if len(batch)>=32 or not pending:
                        with db:db.executemany('UPDATE visual_matches SET decision=?,metrics_json=? WHERE run_id=? AND photo_a=? AND photo_b=?',batch)
                        batch=[]
                    details['verified_processed']=completed;status()
        if stop():stage='stopped';status(True);return
        feature.cache_clear()
        stage='grouping';status(True);adjacent=collections.defaultdict(set)
        for r in db.execute("SELECT photo_a,photo_b FROM visual_matches WHERE run_id=? AND decision='verified_auto'",(run,)):
            adjacent[r[0]].add(r[1]);adjacent[r[1]].add(r[0])
        remaining=set(hashes);groups=[]
        rank=lambda i:((byid[i]['artimuse_score'] or 0),(byid[i]['width_px'] or 0)*(byid[i]['height_px'] or 0),-i)
        for representative in sorted(hashes,key=rank,reverse=True):
            if representative not in remaining:continue
            members=[representative]+sorted(adjacent[representative]&remaining-{representative})
            remaining.difference_update(members)
            if len(members)>1:groups.append((len(groups)+1,representative,members))
        with db:
            db.execute('DELETE FROM visual_group_members WHERE run_id=?',(run,));db.execute('DELETE FROM visual_groups WHERE run_id=?',(run,))
            for ident,representative,members in groups:
                db.execute('INSERT INTO visual_groups VALUES(?,?,?,?)',(run,ident,representative,len(members)))
                db.executemany('INSERT INTO visual_group_members VALUES(?,?,?)',[(run,ident,i) for i in members])
        stage='complete_with_errors' if errors else 'complete'
        details.update(groups=len(groups),grouped_photos=sum(len(g[2]) for g in groups),finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        with db:db.execute('UPDATE visual_hash_runs SET summary_json=? WHERE run_id=?',(json.dumps(dict(state=stage,**details)),run))
        (folder/'summary.json').write_text(json.dumps(dict(run_id=run,state=stage,**details),indent=2));status(True)
    except BaseException as error:
        stage='error';details['error']=str(error);status(True);raise
    finally:db.close();lock.close()

if __name__=='__main__':main()
