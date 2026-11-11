"""Resumable four-slot local hierarchical labeling; preserves existing photos columns."""
import argparse,base64,concurrent.futures,datetime,json,msvcrt,os,shutil,socket,sqlite3,subprocess,time,urllib.request
from pathlib import Path
from portal_config import catalog_paths

APP=Path(__file__).resolve().parent
OUT=APP/'hierarchy_run'
DB,_=catalog_paths()
MODEL='qwen3.5:4b'
POLICY='hierarchy-v3'
COMPATIBLE_PREVIOUS_POLICY='hierarchy-v2'
URL='http://127.0.0.1:11435'
TAXONOMY={
 'scenery':['mountains','lake_river','coast_beach','forest','waterfall','flowers_garden','cityscape','architecture','countryside','snow_ice','other_scenery'],
 'wildlife':['birds','mammals','marine_life','reptiles_amphibians','insects_invertebrates','other_wildlife'],
 'domestic_animals':['dogs','cats','horses','livestock','pet_birds','other_pets'],
 'people':['individual_portrait','couple','family_group','baby_child','wedding','candid','sports_action','other_people'],
 'vehicles':['aircraft','road_vehicles','rail','watercraft','other_vehicles'],
 'other':['food','interiors','objects','documents_screenshots','artwork','other'],
 'uncertain':['uncertain']}
SCHEMA={'type':'object','properties':{
 'primary_category':{'type':'string','enum':list(TAXONOMY)},
 'subcategory':{'type':'string','enum':[s for group in TAXONOMY.values() for s in group]},
 'subject_tags':{'type':'array','items':{'type':'string'},'minItems':1,'maxItems':8},
 'scene_tags':{'type':'array','items':{'type':'string'},'maxItems':6},
 'description':{'type':'string'},'animal_present':{'type':'boolean'}},
 'required':['primary_category','subcategory','subject_tags','scene_tags','description','animal_present'],'additionalProperties':False}
PROMPT='''Describe and categorize this photograph using ONLY visible evidence, without guessing names, locations, dates or identities. Return the requested JSON.
Choose the main photographic subject as primary_category and one matching subcategory from this hierarchy: '''+json.dumps(TAXONOMY)+'''.
Use wildlife or domestic_animals ONLY when a real animal is visible. Animal prints, statues, toys, food and eggs alone do not count as live animals. People holding pets may use domestic_animals, with people in subject_tags. Use people for babies, weddings, and human portraits. Scenery includes landscapes, city views and photographs featuring architecture; a mundane close-up of house parts or a renovation defect belongs to other/objects, rooms to other/interiors. Artwork and screenshots belong to other. Use uncertain when the main subject is unclear.
People holding or posing with prominent pets MUST use domestic_animals as primary_category (e.g. dogs), while retaining person or people in subject_tags. A close-up of a hand holding an object, without a visible person's face or body as the main subject, is other/objects, not a portrait. Use family_group for multiple people when people is the main category; individual_portrait means one person.
When a vehicle is the main subject, use vehicles/aircraft for airplanes, jets, helicopters and aircraft formations, vehicles/road_vehicles for cars, motorcycles, trucks and buses, vehicles/rail for trains and trams, vehicles/watercraft for boats and ships, or vehicles/other_vehicles if the type is unclear. Vehicles merely incidental to a landscape or city view do not make it a vehicle photo.
subject_tags: 1-8 short concrete visible subjects, each 1-3 words (e.g. person, dog, mountain, food); use canonical general nouns alongside specific species when useful, omit clothing details and colors. Multiple subjects allowed. scene_tags: 0-6 short visible setting/lighting/activity tags (e.g. indoors, outdoors, beach, snow, sunset, posing); omit uncertain details. Use lowercase English words with spaces, not underscores. description: one factual sentence, at most 25 words. Do not judge beauty or image quality. animal_present means an actual animal is visible.'''

def api(path,body=None,timeout=300):
    req=urllib.request.Request(URL+path,None if body is None else json.dumps(body).encode(),{'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=timeout) as r:return json.load(r)

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def infer(row):
    start=time.monotonic()
    for attempt in range(3):
        try:
            path=Path(row['thumbnail_path']);st=path.stat()
            body=dict(model=MODEL,messages=[dict(role='user',content=PROMPT,images=[base64.b64encode(path.read_bytes()).decode()])],format=SCHEMA,think=False,stream=False,keep_alive='15m',options=dict(temperature=0,num_ctx=4096,num_predict=280))
            response=api('/api/chat',body)
            if response.get('done_reason')!='stop':raise ValueError('Truncated model response')
            data=json.loads(response['message']['content'])
            if data['subcategory'] not in TAXONOMY[data['primary_category']]:raise ValueError('Inconsistent category hierarchy')
            if not isinstance(data['animal_present'],bool):raise ValueError('Invalid animal flag')
            if data['primary_category'] in ['wildlife','domestic_animals'] and not data['animal_present']:raise ValueError('Animal category without visible animal')
            for field,maximum in [('subject_tags',8),('scene_tags',6)]:
                tags=data[field]
                if not isinstance(tags,list) or len(tags)>maximum or any(not isinstance(t,str) or not t.strip() or len(t)>100 for t in tags):raise ValueError('Invalid tags')
                data[field]=list(dict.fromkeys(' '.join(t.lower().replace('_',' ').replace('-',' ').strip().split()) for t in tags))
                data[field]=list(dict.fromkeys({'indoor':'indoors','outdoor':'outdoors','closeup':'close up'}.get(t,t) for t in data[field]))
            if not data['subject_tags'] or not isinstance(data['description'],str) or not data['description'].strip():raise ValueError('Empty subjects or description')
            return dict(row=row,status='ok',data=data,raw=response['message']['content'],size=st.st_size,mtime=st.st_mtime_ns,seconds=time.monotonic()-start,error=None)
        except Exception as error:
            if attempt<2:time.sleep(1+attempt)
            else:return dict(row=row,status='failed',error=f'{type(error).__name__}: {error}',seconds=time.monotonic()-start)

def schema(db):
    db.executescript('''
    CREATE TABLE IF NOT EXISTS photo_classifications (
      photo_id INTEGER PRIMARY KEY REFERENCES photos(id), primary_category TEXT, subcategory TEXT,
      subject_tags_json TEXT, scene_tags_json TEXT, description TEXT, animal_present INTEGER,
      model TEXT NOT NULL, policy TEXT NOT NULL, status TEXT NOT NULL, error TEXT,
      classified_at TEXT, thumbnail_size INTEGER, thumbnail_mtime_ns INTEGER,
      inference_seconds REAL, raw_json TEXT);
    CREATE INDEX IF NOT EXISTS idx_pc_category ON photo_classifications(primary_category,subcategory,status);
    CREATE INDEX IF NOT EXISTS idx_pc_status ON photo_classifications(status);
    CREATE TABLE IF NOT EXISTS photo_classification_tags (
      photo_id INTEGER REFERENCES photo_classifications(photo_id), kind TEXT NOT NULL,
      tag TEXT NOT NULL, PRIMARY KEY(photo_id,kind,tag));
    CREATE INDEX IF NOT EXISTS idx_pct_tag ON photo_classification_tags(kind,tag,photo_id);
    CREATE VIEW IF NOT EXISTS photos_hierarchical AS
      SELECT p.*,c.primary_category,c.subcategory,c.subject_tags_json,c.scene_tags_json,
      c.description AS classification_description,c.animal_present,
      c.status AS classification_status,c.model AS classification_model
      FROM photos p LEFT JOIN photo_classifications c ON p.id=c.photo_id;
    ''');db.commit()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pilot',action='store_true');args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    (OUT/'policy.json').write_text(json.dumps(dict(model=MODEL,policy=POLICY,taxonomy=TAXONOMY,schema=SCHEMA,prompt=PROMPT),indent=2),encoding='utf-8')
    lock=(OUT/'run.lock').open('a+b');lock.seek(0)
    if not lock.read(1):lock.write(b'0');lock.flush()
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    (OUT/'stop.request').unlink(missing_ok=True)
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1',11435))==0:raise RuntimeError('Port 11435 is occupied; refusing to modify that service')
    db=sqlite3.connect(str(DB),timeout=60);db.row_factory=sqlite3.Row;db.execute('PRAGMA foreign_keys=ON');schema(db)
    # Existing successful v2 labels remain usable after the Vehicles migration.
    rows=[dict(r) for r in db.execute("SELECT p.id,p.thumbnail_path FROM photos p LEFT JOIN photo_classifications c ON p.id=c.photo_id WHERE p.thumbnail_status='ok' AND (c.photo_id IS NULL OR c.status!='ok' OR c.model!=? OR c.policy NOT IN (?,?) OR (c.primary_category='other' AND c.subcategory='vehicles')) ORDER BY p.id",(MODEL,POLICY,COMPATIBLE_PREVIOUS_POLICY))]
    if args.pilot:rows=rows[:12]
    env=os.environ.copy();env.update(OLLAMA_HOST='127.0.0.1:11435',OLLAMA_NUM_PARALLEL='4',OLLAMA_MAX_LOADED_MODELS='1')
    stdout=(OUT/'ollama-out.log').open('a');stderr=(OUT/'ollama-err.log').open('a')
    server=subprocess.Popen([shutil.which('ollama'),'serve'],env=env,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)
    started=time.monotonic();completed=failed=0;state='starting'
    def status():
        counts=dict(db.execute('SELECT status,COUNT(*) FROM photo_classifications GROUP BY status'))
        eligible=db.execute("SELECT COUNT(*) FROM photos WHERE thumbnail_status='ok'").fetchone()[0]
        value=dict(state=state,pid=os.getpid(),server_pid=server.pid,model=MODEL,policy=POLICY,slots=4,pilot=args.pilot,updated_at=now(),run_total=len(rows),run_completed=completed,run_failed=failed,catalog_classifications=counts,eligible_photos=eligible,percent=round(100*counts.get('ok',0)/eligible,3) if eligible else 100,elapsed_seconds=round(time.monotonic()-started,1))
        temp=OUT/'status.tmp';temp.write_text(json.dumps(value,indent=2));os.replace(temp,OUT/'status.json')
    status()
    try:
        deadline=time.monotonic()+25
        while True:
            try:api('/api/version',timeout=1);break
            except Exception:
                if server.poll() is not None or time.monotonic()>deadline:raise RuntimeError('Ollama server failed to start')
                time.sleep(.2)
        state='running';status();iterator=iter(rows)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            pending={pool.submit(infer,row) for row in [next(iterator,None) for _ in range(4)] if row}
            while pending:
                done,pending=concurrent.futures.wait(pending,timeout=15,return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    r=future.result();ident=r['row']['id'];d=r.get('data',{})
                    with db:
                        db.execute('''INSERT OR REPLACE INTO photo_classifications VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                         (ident,d.get('primary_category'),d.get('subcategory'),json.dumps(d.get('subject_tags',[])),json.dumps(d.get('scene_tags',[])),d.get('description'),d.get('animal_present'),MODEL,POLICY,r['status'],r['error'],now(),r.get('size'),r.get('mtime'),r['seconds'],r.get('raw')))
                        db.execute('DELETE FROM photo_classification_tags WHERE photo_id=?',(ident,))
                        for field,kind in [('subject_tags','subject'),('scene_tags','scene')]:
                            db.executemany('INSERT INTO photo_classification_tags VALUES(?,?,?)',[(ident,kind,t) for t in d.get(field,[])])
                    completed+=1;failed+=r['status']=='failed'
                    if completed%100==0:print(json.dumps(dict(completed=completed,failed=failed,total=len(rows),elapsed_seconds=round(time.monotonic()-started,1))),flush=True)
                    if args.pilot:print(json.dumps(dict(id=ident,**d,status=r['status'],error=r['error'])),flush=True)
                    if not (OUT/'stop.request').exists():
                        row=next(iterator,None)
                        if row:pending.add(pool.submit(infer,row))
                status()
                if server.poll() is not None:raise RuntimeError('Ollama process exited unexpectedly')
        state='stopped' if (OUT/'stop.request').exists() else 'complete_with_errors' if failed else 'complete';status()
    except BaseException:
        state='error';status();raise
    finally:
        try:api('/api/generate',dict(model=MODEL,keep_alive=0),timeout=30)
        except Exception:pass
        server.terminate()
        try:server.wait(timeout=10)
        except subprocess.TimeoutExpired:server.kill();server.wait(timeout=10)
        stdout.close();stderr.close();db.close();lock.close()

if __name__=='__main__':main()
