"""Local photo query portal. Standard library + Pillow; catalog is read-only."""
import argparse,collections,csv,datetime,hashlib,http.server,io,json,mimetypes,os,random,re,sqlite3,subprocess,threading,time,urllib.parse
from bisect import bisect_left,bisect_right
from math import isfinite
from pathlib import Path
from PIL import Image
from tag_filters import canonical,normalized
from classify_hierarchy import TAXONOMY
from portal_config import catalog_paths,folder_key,load_config,is_visible
from functools import lru_cache

APP=Path(__file__).resolve().parent
DB,ROOTS=catalog_paths()
ROOT_KEYS=tuple(folder_key(str(root)) for root in ROOTS)
LABELS={'scenery':'Scenery','wildlife':'Wildlife','people':'People','domestic_animals':'Domestic animals','vehicles':'Vehicles','other':'Other', 'uncertain':'Uncertain'}
CAMERAS="""((UPPER(camera_make)='CANON' AND UPPER(camera_model) LIKE 'CANON EOS%') OR
 (UPPER(camera_make) LIKE 'NIKON%' AND (UPPER(camera_model) GLOB 'NIKON D[0-9]*' OR UPPER(camera_model) LIKE 'NIKON Z%')) OR
 (UPPER(camera_make)='SONY' AND (UPPER(camera_model) LIKE 'ILCE-%' OR UPPER(camera_model) LIKE 'NEX-%')) OR
 (UPPER(camera_make) LIKE 'OLYMPUS%' AND UPPER(camera_model)='E-1'))"""
FIELDS='id,root_folder,folder_path,file_name,relative_path,full_path,thumbnail_path,category,category_margin,artimuse_score,artimuse_status,fgaesq_score,fgaesq_rank,category_rank,folder_rank,camera_make,camera_model,lens_model,focal_length_mm,aperture_f_number,iso,exposure_time_description,date_taken,width_px,height_px,thumbnail_status'
# TEMPORARY: only fully labeled photos enter gallery views. Keep this gate centralized.
HIERARCHY_SOURCE="""(SELECT p.*,c.primary_category,c.subcategory,c.subject_tags_json,c.scene_tags_json,
 c.description AS classification_description,c.animal_present,c.model AS classification_model,
 c.status AS classification_status FROM photos p JOIN photo_classifications c ON c.photo_id=p.id
 WHERE c.status='ok' AND p.artimuse_status='ok' AND p.artimuse_score IS NOT NULL AND portal_visible(p.folder_path)) AS photos"""
GALLERY_FIELDS=FIELDS.replace('category,category_margin','primary_category AS category,NULL AS category_margin').replace('category_rank,folder_rank','NULL AS category_rank,NULL AS folder_rank')+',primary_category,subcategory,subject_tags_json,scene_tags_json,classification_description,animal_present,classification_model,classification_status,category AS legacy_category'
TAG_LOCK=threading.Lock()
TAG_CACHE={'expires':0}

def tag_catalog():
    with TAG_LOCK:
        config_key=tuple(load_config()['hidden_folders'])
        if time.monotonic()<TAG_CACHE['expires'] and TAG_CACHE.get('config_key')==config_key:return TAG_CACHE['value']
        with connect() as c:
            c.execute('BEGIN')  # One consistent snapshot while the labeling job adds tags.
            raw=[dict(r) for r in c.execute("SELECT t.kind,t.tag FROM photo_classification_tags t JOIN photo_classifications pc ON pc.photo_id=t.photo_id JOIN photos p ON p.id=t.photo_id WHERE pc.status='ok' AND portal_visible(p.folder_path) AND p.thumbnail_status='ok' AND p.analysis_status='ok' GROUP BY t.kind,t.tag")]
            vocabulary={normalized(r['tag']) for r in raw}
            variants={}
            for r in raw:
                key=(r['kind'],canonical(r['tag'],vocabulary))
                variants.setdefault(key,set()).add(r['tag'])
            c.create_function('canonical_tag',1,lambda tag:canonical(tag,vocabulary),deterministic=True)
            counts=c.execute("SELECT t.kind,canonical_tag(t.tag) AS label,COUNT(DISTINCT t.photo_id) AS n FROM photo_classification_tags t JOIN photo_classifications pc ON pc.photo_id=t.photo_id JOIN photos p ON p.id=t.photo_id WHERE pc.status='ok' AND portal_visible(p.folder_path) AND p.thumbnail_status='ok' AND p.analysis_status='ok' GROUP BY t.kind,label").fetchall()
            value={kind:[] for kind in ['subject','scene']}
            for r in counts:value[r['kind']].append(dict(tag=r['label'],count=r['n'],variants=sorted(variants[(r['kind'],r['label'])])))
            for entries in value.values():entries.sort(key=lambda r:(-r['count'],r['tag']))
        TAG_CACHE.update(expires=time.monotonic()+30,value=value,config_key=config_key,lookup={kind:{v:r['tag'] for r in value[kind] for v in r['variants']} for kind in value})
        return value

def tag_suggestions(kind,query=''):
    if kind not in ('subject','scene'):raise ValueError('Unknown tag kind')
    query=normalized(query)
    entries=tag_catalog()[kind]
    matching=[r for r in entries if not query or query in r['tag'] or any(query in normalized(v) for v in r['variants'])]
    matching.sort(key=lambda r:(0 if r['tag']==query else 1 if r['tag'].startswith(query) else 2,-r['count'],r['tag']))
    return dict(tags=matching[:30],total=len(matching),count_scope='Successfully labeled collection; other gallery filters are not applied')

CACHE_LOCK=threading.Lock()

def connect():
    c=sqlite3.connect(DB.as_uri()+'?mode=ro',uri=True,timeout=10)
    hidden=tuple(load_config()['hidden_folders'])
    @lru_cache(maxsize=8192)
    def visible(folder):return int(is_visible(folder or '',hidden))
    c.create_function('portal_visible',1,visible,deterministic=True)
    c.row_factory=sqlite3.Row;c.execute('PRAGMA query_only=ON')
    return c

def clean(value):
    if isinstance(value,bytes):return f'[binary data: {len(value)} bytes]'
    if isinstance(value,float) and not __import__('math').isfinite(value):return None
    return value

def score_distribution(c):
    """Stable reference: all successfully scored, visible catalog thumbnails."""
    return [r[0] for r in c.execute("""SELECT artimuse_score FROM photos
        WHERE artimuse_status='ok' AND thumbnail_status='ok'
          AND artimuse_score BETWEEN 0 AND 100 AND portal_visible(folder_path)
        ORDER BY artimuse_score""")]

def score_percentile(scores,value):
    if not scores or not isinstance(value,(float,int)) or not isfinite(value):return None
    return 100*bisect_right(scores,value)/len(scores)

def photo(row,scores):
    p={k:clean(v) for k,v in dict(row).items()};ident=p['id']
    p['artimuse_percentile']=score_percentile(scores,p.get('artimuse_score'))

    if p.get('classification_status')=='ok':
        p['category']=p['primary_category'];p['category_margin']=None
        p['subject_tags']=json.loads(p.get('subject_tags_json') or '[]');p['scene_tags']=json.loads(p.get('scene_tags_json') or '[]')
        catalog=tag_catalog()
        for kind,field in [('subject','subject_tags'),('scene','scene_tags')]:
            lookup=TAG_CACHE['lookup'][kind]
            p[field]=list(dict.fromkeys(lookup.get(v,normalized(v)) for v in p[field]))
    p.update(thumbnail_url=f'/media/thumbnail/{ident}',original_url=f'/media/original/{ident}')
    return p

def bounded_int(value,default,low,high):
    try:return max(low,min(high,int(value)))
    except (TypeError,ValueError):return default

def metadata():
    with connect() as c:
        counts=dict(c.execute('SELECT artimuse_status,COUNT(*) FROM photos WHERE portal_visible(folder_path) GROUP BY artimuse_status'))
        labeled=c.execute('SELECT COUNT(*) FROM '+HIERARCHY_SOURCE+" WHERE analysis_status='ok' AND thumbnail_status='ok'").fetchone()[0]
        cameras=[dict(r) for r in c.execute('SELECT camera_model,COUNT(*) AS count FROM '+HIERARCHY_SOURCE+" WHERE camera_model IS NOT NULL GROUP BY camera_model ORDER BY count DESC")]
        categories=[dict(r) for r in c.execute('SELECT primary_category AS category,COUNT(*) AS count FROM '+HIERARCHY_SOURCE+" WHERE analysis_status='ok' AND thumbnail_status='ok' GROUP BY primary_category")]
        subcategories=[dict(r) for r in c.execute('SELECT primary_category AS category,subcategory,COUNT(*) AS count FROM '+HIERARCHY_SOURCE+" WHERE analysis_status='ok' AND thumbnail_status='ok' GROUP BY primary_category,subcategory")]
        fields=[dict(r) for r in c.execute('PRAGMA table_info(photos_hierarchical)')]
        active=c.execute("SELECT model FROM artimuse_runs WHERE status='complete' ORDER BY finished_at DESC LIMIT 1").fetchone()
        visual_info=visual_catalog()
        return dict(visual=dict(run_id=visual_info['run_id'],groups=visual_info['groups']),score_model=active['model'].rsplit('/',1)[-1] if active else 'ArtiMuse',score_display='percentile',score_scale=[0,100],app='photo-portal-v1',database=str(DB),total=sum(counts.values()),scored=counts.get('ok',0),labeled=labeled,counts=counts,cameras=cameras,categories=categories,subcategories=subcategories,taxonomy=TAXONOMY,labels=LABELS,roots=[str(r) for r in ROOTS],schema=fields,labeled_only=True)

def fingerprint(row,cache,pending):
    p=Path(row['thumbnail_path']);st=p.stat();key=str(p)
    cached=cache.get(key)
    if cached and cached[0]==st.st_size and cached[1]==st.st_mtime_ns:return cached[2]
    with Image.open(p) as source:
        im=source.convert('RGB');digest=hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()
    cache[key]=(st.st_size,st.st_mtime_ns,digest);pending.append((key,st.st_size,st.st_mtime_ns,digest))
    return digest

def keep_best_labels(rows,tagsets):
    """Highest score per folder + hierarchy + unordered canonical subject/scene sets."""
    lookup={kind:{v:r['tag'] for r in tagsets[kind] for v in r['variants']} for kind in ['subject','scene']}
    def key(row):
        sets=[]
        for kind,field in [('subject','subject_tags_json'),('scene','scene_tags_json')]:
            tags=json.loads(row.get(field) or '[]')
            sets.append(tuple(sorted({lookup[kind].get(t,normalized(t)) for t in tags})))
        return (row['folder_path'].replace('/','\\').rstrip('\\').casefold(),row['category'],row.get('subcategory'),*sets)
    def rank(row):return (row['artimuse_score'] if row['artimuse_score'] is not None else float('-inf'),-row['id'])
    winners={}
    for row in rows:
        group=key(row)
        if group not in winners or rank(row)>rank(winners[group]):winners[group]=row
    ids={row['id'] for row in winners.values()}
    return [row for row in rows if row['id'] in ids]

def visual_catalog():
    """Use the latest completed full run; never a pilot or partially built run."""
    with connect() as c:
        run=None
        for r in c.execute("SELECT run_id,summary_json FROM visual_hash_runs WHERE scope='full' ORDER BY created_at DESC"):
            if json.loads(r['summary_json']).get('state') in ('complete','complete_with_errors'):
                run=r['run_id'];break
        if not run:return dict(run_id=None,members={},sizes={},groups=0)
        members={r['photo_id']:r['group_id'] for r in c.execute('SELECT m.photo_id,m.group_id FROM visual_group_members m JOIN '+HIERARCHY_SOURCE+" ON photos.id=m.photo_id WHERE m.run_id=? AND thumbnail_status='ok' AND analysis_status='ok'",(run,))}
        sizes=dict(collections.Counter(members.values()))
        return dict(run_id=run,members=members,sizes=sizes,groups=sum(n>1 for n in sizes.values()))

def attach_visual(rows,catalog):
    for row in rows:
        group=catalog['members'].get(row['id'])
        row.update(visual_group_id=group,visual_run_id=catalog['run_id'],visual_group_size=catalog['sizes'].get(group,1))
    return rows

def keep_visual_representatives(rows):
    winners={}
    def rank(row):return (row['artimuse_score'] if row['artimuse_score'] is not None else float('-inf'),(row.get('width_px') or 0)*(row.get('height_px') or 0),-row['id'])
    for row in rows:
        key=('group',row['visual_group_id']) if row.get('visual_group_id') is not None else ('photo',row['id'])
        if key not in winners or rank(row)>rank(winners[key]):winners[key]=row
    ids={r['id'] for r in winners.values()}
    return [r for r in rows if r['id'] in ids]

def versions(ident):
    catalog=visual_catalog()
    with connect() as c:
        c.execute('BEGIN')
        scores=score_distribution(c)
        source=c.execute('SELECT '+GALLERY_FIELDS+' FROM '+HIERARCHY_SOURCE+" WHERE id=? AND thumbnail_status='ok' AND analysis_status='ok'",(ident,)).fetchone()
        if source is None:raise ValueError('Photo not available in the visible labeled collection.')
        group=catalog['members'].get(ident)
        if group is None:rows=[dict(source)]
        else:rows=[dict(r) for r in c.execute('SELECT '+GALLERY_FIELDS+' FROM '+HIERARCHY_SOURCE+' JOIN visual_group_members m ON m.photo_id=photos.id WHERE m.run_id=? AND m.group_id=? AND thumbnail_status=\'ok\' AND analysis_status=\'ok\' ORDER BY artimuse_score DESC,(width_px*height_px) DESC,id',(catalog['run_id'],group))]
    return dict(photos=[photo(r,scores) for r in attach_visual(rows,catalog)],count=len(rows),source_id=ident,group_id=group,run_id=catalog['run_id'])

def shoot_key(folder):
    """A shoot is the first folder below a photo root, including all descendants."""
    for root in ROOT_KEYS:
        prefix=root+'\\'
        if folder.startswith(prefix):
            return prefix+folder[len(prefix):].split('\\',1)[0]
    return folder


def gallery(options):
    started=time.monotonic();where=["analysis_status='ok'","thumbnail_status='ok'"];params=[]
    if options.get('camera_only',True):where+= ["metadata_status='ok'",CAMERAS]
    elif options.get('exif_only',False):where.append("metadata_status='ok'")
    categories=[x for x in options.get('categories',list(LABELS)) if x in LABELS]
    if not categories:categories=list(LABELS)
    where.append('primary_category IN ('+','.join('?' for _ in categories)+')');params.extend(categories)
    subfilters=options.get('subcategory_by_category') or {}
    if not isinstance(subfilters,dict):raise ValueError('Invalid subcategories')
    for cat,sub in subfilters.items():
        if cat not in LABELS or sub not in TAXONOMY[cat]:raise ValueError('Unknown subcategory')
        where.append('(primary_category!=? OR subcategory=?)');params.extend([cat,sub])
    tagsets=tag_catalog()
    for kind in ['subject','scene']:
        chosen=options.get(kind+'_tags') or []
        if not isinstance(chosen,list) or len(chosen)>12 or any(not isinstance(x,str) or len(x)>100 for x in chosen):raise ValueError('Select at most 12 tags per group')
        lookup={r['tag']:r['variants'] for r in tagsets[kind]}
        vocabulary={normalized(v) for r in tagsets[kind] for v in r['variants']}
        groups=[]
        for tag in dict.fromkeys(chosen):
            key=canonical(tag,vocabulary)
            variants=lookup.get(key,[])
            if not variants:groups.append('0');continue
            groups.append("EXISTS (SELECT 1 FROM photo_classification_tags t WHERE t.photo_id=photos.id AND t.kind=? AND t.tag IN ("+','.join('?' for _ in variants)+'))')
            params.extend([kind,*variants])
        mode=options.get(kind+'_match','any')
        if mode not in ('any','all'):raise ValueError('Unknown tag match mode')
        if groups:where.append('('+(' AND ' if mode=='all' else ' OR ').join(groups)+')')
    root=options.get('root')
    if root:
        if root not in [str(r) for r in ROOTS]:raise ValueError('Unknown photo root')
        where.append('root_folder=?');params.append(root)
    if options.get('camera_model'):where.append('camera_model=?');params.append(str(options['camera_model']))
    if options.get('folder'):
        where.append('folder_path LIKE ? ESCAPE \'\\\'');value=str(options['folder']).replace('\\','\\\\').replace('%','\\%').replace('_','\\_');params.append('%'+value+'%')
    for name,column in [('min_score','artimuse_score'),('min_focal','focal_length_mm'),('min_iso','iso')]:
        if options.get(name) not in (None,''):
            where.append(column+'>=?');params.append(float(options[name]))
    for name,column in [('max_iso','iso'),('max_focal','focal_length_mm'),('max_aperture','aperture_f_number')]:
        if options.get(name) not in (None,''):
            where.append(column+'<=?');params.append(float(options[name]))
    sort={'score':'artimuse_score DESC,id','newest':'date_taken DESC,id','oldest':'date_taken ASC,id','iso':'iso ASC,artimuse_score DESC,id'}.get(options.get('sort'),'artimuse_score DESC,id')
    with connect() as c:
        c.execute('BEGIN')
        scores=score_distribution(c)
        if options.get('min_percentile') not in (None,''):
            try:minimum=float(options['min_percentile'])
            except (TypeError,ValueError):raise ValueError('Minimum percentile must be between 0 and 100')
            if not 0<=minimum<=100:raise ValueError('Minimum percentile must be between 0 and 100')
            if scores:
                # Translate to a raw-score cutoff. Equal scores always pass or fail together.
                index=bisect_left(range(1,len(scores)+1),minimum,key=lambda rank:100*rank/len(scores))
                where.append('artimuse_score>=?');params.append(scores[index])
            else:where.append('0')
        sql='SELECT '+GALLERY_FIELDS+' FROM '+HIERARCHY_SOURCE+' WHERE '+' AND '.join(where)+' ORDER BY '+sort
        rows=[dict(r) for r in c.execute(sql,params)]
    visual_data=visual_catalog()
    attach_visual(rows,visual_data)
    before_visual=len(rows)
    if options.get('visual_grouping',True):rows=keep_visual_representatives(rows)
    visual_duplicates=before_visual-len(rows)
    before_label_group=len(rows)
    if options.get('best_labels_per_folder',False):rows=keep_best_labels(rows,tagsets)
    label_duplicates=before_label_group-len(rows)
    available=collections.Counter(r['category'] for r in rows)
    per_category=bounded_int(options.get('per_category'),50,1,2000)
    cap=bounded_int(options.get('folder_cap'),0,0,100)
    explore=bounded_int(options.get('explore_percent'),0,0,100)
    rng=random.Random()
    def candidates(items):
        shuffled=items.copy();rng.shuffle(shuffled)
        ordered_iter=iter(items);random_iter=iter(shuffled);yielded=set()
        while len(yielded)<len(items):
            source=random_iter if rng.random()<explore/100 else ordered_iter
            for row in source:
                if row['id'] not in yielded:
                    yielded.add(row['id']);yield row;break
    pools={cat:candidates([r for r in rows if r['category']==cat]) for cat in categories}
    selected=[];folder_counts=collections.Counter();shoot_counts=collections.Counter();cat_counts=collections.Counter();seen=set();duplicates=0
    cache={};cache_pending=[]
    deduplicate=bool(options.get('deduplicate',True))
    if deduplicate:
        with CACHE_LOCK,sqlite3.connect(APP/'preview_hashes.sqlite') as cache_db:
            cache_db.execute('CREATE TABLE IF NOT EXISTS hashes(path TEXT PRIMARY KEY,size INTEGER,mtime_ns INTEGER,hash TEXT)')
            cache={r[0]:r[1:] for r in cache_db.execute('SELECT path,size,mtime_ns,hash FROM hashes')}
    # Round-robin categories share the gallery-wide shoot cap.
    active=list(categories)
    while active:
        next_active=[]
        for cat in active:
            if cat_counts[cat]>=per_category:continue
            for row in pools[cat]:
                folder=folder_key(row['folder_path'])
                shoot=shoot_key(folder) if cap else None
                if cap and shoot_counts[shoot]>=cap:continue
                if deduplicate:
                    try:digest=fingerprint(row,cache,cache_pending)
                    except (OSError,ValueError):continue
                    if digest in seen:duplicates+=1;continue
                    seen.add(digest)
                selected.append(row);folder_counts[folder]+=1;cat_counts[cat]+=1
                if shoot is not None:shoot_counts[shoot]+=1
                next_active.append(cat);break
        active=next_active
    if cache_pending:
        with CACHE_LOCK,sqlite3.connect(APP/'preview_hashes.sqlite',timeout=20) as cache_db:
            cache_db.executemany('INSERT OR REPLACE INTO hashes VALUES(?,?,?,?)',cache_pending)
    order={r['id']:i for i,r in enumerate(rows)}
    selected.sort(key=lambda r:(categories.index(r['category']),order[r['id']]))
    # Preserve the selected order within each category from the ordered SQL pool.
    result=[];ranks=collections.Counter()
    for row in selected:
        ranks[row['category']]+=1;p=photo(row,scores);p['selection_rank']=ranks[row['category']];result.append(p)
    return dict(photos=result,count=len(result),available=sum(available.values()),available_by_category=dict(available),selected_by_category=dict(cat_counts),folders=len(folder_counts),duplicates_skipped=duplicates,label_duplicates_skipped=label_duplicates,visual_duplicates_skipped=visual_duplicates,visual_run_id=visual_data['run_id'],matching_before_visual_group=before_visual,matching_before_label_group=before_label_group,seconds=round(time.monotonic()-started,3),sql=sql,params=params,folder_cap=cap)

def authorizer(action,arg1,arg2,db,trigger):
    if action in (sqlite3.SQLITE_SELECT,sqlite3.SQLITE_READ,sqlite3.SQLITE_RECURSIVE):return sqlite3.SQLITE_OK
    if action==sqlite3.SQLITE_FUNCTION:
        return sqlite3.SQLITE_DENY if str(arg2 or arg1).lower() in {'load_extension','readfile','writefile','edit','eval','fts3_tokenizer'} else sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY

def reveal_photo(options):
    if os.name!='nt':raise ValueError('Open in File Explorer is available on Windows.')
    ident=options.get('id')
    if isinstance(ident,bool) or not isinstance(ident,int):raise ValueError('A catalog photo ID is required.')
    with connect() as c:row=c.execute('SELECT full_path FROM photos WHERE id=? AND portal_visible(folder_path)',(ident,)).fetchone()
    if row is None:raise ValueError('Photo not found in the catalog.')
    path=Path(row['full_path']).resolve()
    if not any(path.is_relative_to(root) for root in ROOTS):raise ValueError('Photo is outside the configured photo folders.')
    if not path.is_file():raise ValueError('The original photo is no longer available.')
    explorer=Path(os.environ.get('WINDIR','C:/Windows'))/'explorer.exe'
    subprocess.Popen(f'"{explorer}" /select,"{path}"',shell=False)
    return dict(opened=True,folder=str(path.parent),file_name=path.name)

def custom_query(options):
    started=time.monotonic();sql=str(options.get('sql','')).strip()
    if not sql:raise ValueError('Enter a SELECT query first.')
    if len(sql)>100_000:raise ValueError('SQL is too long.')
    limit=bounded_int(options.get('limit'),500,1,2000)
    c=connect()
    try:
        c.execute('BEGIN')
        scores=score_distribution(c)
        c.create_function('artimuse_percentile',1,lambda value:score_percentile(scores,value),deterministic=True)
        c.setlimit(sqlite3.SQLITE_LIMIT_LENGTH,2_000_000);c.setlimit(sqlite3.SQLITE_LIMIT_COLUMN,200)
        c.set_authorizer(authorizer);c.set_progress_handler(lambda:1 if time.monotonic()-started>6 else 0,2000)
        cursor=c.execute(sql)
        if not cursor.description:raise ValueError('Use a SELECT query that returns rows.')
        columns=[d[0] for d in cursor.description];raw=cursor.fetchmany(limit+1);truncated=len(raw)>limit
        values=[[clean(v) for v in r] for r in raw[:limit]]
    finally:c.close()
    records=[dict(zip(columns,row)) for row in values];ids=[];paths=[]
    for row in records:
        ident=row.get('id',row.get('photo_id'))
        if isinstance(ident,int):ids.append(ident)
        elif row.get('full_path'):paths.append(str(row['full_path']))
    mapped={}
    with connect() as c:
        if ids:
            for r in c.execute('SELECT '+GALLERY_FIELDS+' FROM '+HIERARCHY_SOURCE+' WHERE thumbnail_status=\'ok\' AND id IN ('+','.join('?' for _ in ids)+')',ids):mapped[r['id']]=photo(r,scores)
        if paths:
            for r in c.execute('SELECT '+GALLERY_FIELDS+' FROM '+HIERARCHY_SOURCE+' WHERE thumbnail_status=\'ok\' AND full_path IN ('+','.join('?' for _ in paths)+')',paths):mapped[r['full_path']]=photo(r,scores)
    photos=[];seen=set()
    for row in records:
        key=row.get('id',row.get('photo_id',row.get('full_path')));p=mapped.get(key)
        if p and p['id'] not in seen:photos.append(p);seen.add(p['id'])
    attach_visual(photos,visual_catalog())
    return dict(columns=columns,rows=values,count=len(values),truncated=truncated,photos=photos,seconds=round(time.monotonic()-started,3),sql=sql)

class Handler(http.server.BaseHTTPRequestHandler):
    server_version='PhotoPortal/1.0'
    def log_message(self,fmt,*args):print(datetime.datetime.now().isoformat(timespec='seconds'),fmt%args,flush=True)
    def response(self,code,payload,content_type='application/json; charset=utf-8'):
        data=json.dumps(payload,ensure_ascii=False,allow_nan=False).encode('utf-8') if content_type.startswith('application/json') else payload
        self.send_response(code);self.send_header('Content-Type',content_type);self.send_header('Content-Length',str(len(data)));self.send_header('X-Content-Type-Options','nosniff');self.send_header('Cache-Control','no-store');self.end_headers()
        try:self.wfile.write(data)
        except (BrokenPipeError,ConnectionResetError):pass
    def allowed(self):
        host=self.headers.get('Host','').split(':')[0]
        if host not in ('localhost','127.0.0.1'):return False
        origin=self.headers.get('Origin')
        return not origin or origin in (f'http://localhost:{self.server.server_port}',f'http://127.0.0.1:{self.server.server_port}')
    def do_GET(self):
        if not self.allowed():return self.response(403,dict(error='Local access only.'))
        path=urllib.parse.urlparse(self.path).path
        try:
            if path=='/api/health':return self.response(200,dict(score_model='ArtiMuse',score_scale=[0,100],app='photo-portal-v1',database=str(DB)))
            if path=='/api/meta':return self.response(200,metadata())
            if path=='/api/tags':
                query=urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                return self.response(200,tag_suggestions(query.get('kind',['subject'])[0],query.get('q',[''])[0]))
            if path.startswith('/api/versions/'):
                return self.response(200,versions(int(path.rsplit('/',1)[1])))
            if path.startswith('/api/photo/'):
                ident=int(path.rsplit('/',1)[1])
                with connect() as c:
                    c.execute('BEGIN')
                    r=c.execute('SELECT photos.*,primary_category AS new_category FROM '+HIERARCHY_SOURCE+' WHERE id=?',(ident,)).fetchone()
                    if r is None:return self.response(404,dict(error='Photo not found.'))
                    return self.response(200,photo(r,score_distribution(c)))
            media=re.fullmatch(r'/media/(thumbnail|original)/(\d+)',path)
            if media:return self.serve_photo(media.group(1),int(media.group(2)))
            static={'/':'index.html','/index.html':'index.html','/app.js':'app.js','/style.css':'style.css'}
            if path not in static:return self.response(404,dict(error='Not found.'))
            file=APP/static[path]
            content_type={'.html':'text/html','.js':'text/javascript','.css':'text/css'}[file.suffix]
            return self.response(200,file.read_bytes(),content_type+'; charset=utf-8')
        except (OSError,ValueError,sqlite3.Error) as e:return self.response(400,dict(error=str(e)))
    def do_POST(self):
        if not self.allowed():return self.response(403,dict(error='Local access only.'))
        try:
            length=int(self.headers.get('Content-Length',0))
            if not 0<length<=200_000:return self.response(413,dict(error='Request is too large.'))
            options=json.loads(self.rfile.read(length))
            if not isinstance(options,dict):raise ValueError('Request must be a JSON object.')
            if self.path=='/api/gallery':return self.response(200,gallery(options))
            if self.path=='/api/query':return self.response(200,custom_query(options))
            if self.path=='/api/reveal':return self.response(200,reveal_photo(options))
            return self.response(404,dict(error='Not found.'))
        except (ValueError,TypeError,sqlite3.Error,OSError) as e:return self.response(400,dict(error=str(e)))
    def serve_photo(self,kind,ident):
        with connect() as c:r=c.execute('SELECT full_path,thumbnail_path FROM photos WHERE id=? AND portal_visible(folder_path)',(ident,)).fetchone()
        if r is None:return self.response(404,dict(error='Photo not found.'))
        p=Path(r['thumbnail_path'] if kind=='thumbnail' else r['full_path']).resolve()
        if not any(p.is_relative_to(root) for root in ROOTS):return self.response(403,dict(error='Image is outside the photo roots.'))
        st=p.stat();etag=f'"{st.st_size:x}-{st.st_mtime_ns:x}"'
        if self.headers.get('If-None-Match')==etag:self.send_response(304);self.end_headers();return
        start,end=0,st.st_size-1;code=200
        requested=self.headers.get('Range')
        if requested:
            match=re.fullmatch(r'bytes=(\d*)-(\d*)',requested)
            if not match:return self.response(416,dict(error='Unsupported byte range.'))
            if match[1]:start=int(match[1]);end=min(end,int(match[2])) if match[2] else end
            elif match[2]:start=max(0,st.st_size-int(match[2]))
            if not 0<=start<=end<st.st_size:return self.response(416,dict(error='Invalid byte range.'))
            code=206
        self.send_response(code);self.send_header('Content-Type','image/jpeg' if kind=='thumbnail' else mimetypes.guess_type(str(p))[0] or 'application/octet-stream')
        self.send_header('Content-Length',str(end-start+1));self.send_header('Accept-Ranges','bytes');self.send_header('ETag',etag);self.send_header('Cache-Control','private,max-age=300');self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Content-Disposition',"inline; filename*=UTF-8''"+urllib.parse.quote(p.name))
        if code==206:self.send_header('Content-Range',f'bytes {start}-{end}/{st.st_size}')
        self.end_headers()
        try:
            with p.open('rb') as f:
                f.seek(start);remaining=end-start+1
                while remaining:
                    chunk=f.read(min(256*1024,remaining))
                    if not chunk:break
                    self.wfile.write(chunk);remaining-=len(chunk)
        except (BrokenPipeError,ConnectionResetError):pass

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8787);args=parser.parse_args()
    if not DB.is_file():parser.error('Catalog not found. Set database in config.json or run catalog.py.')
    for port in range(args.port,args.port+20):
        try:server=http.server.ThreadingHTTPServer(('127.0.0.1',port),Handler);break
        except OSError:
            if port==args.port+19:raise
    server.daemon_threads=True;url=f'http://127.0.0.1:{server.server_port}/'
    (APP/'server.json').write_text(json.dumps(dict(pid=os.getpid(),url=url,port=server.server_port,database=str(DB)),indent=2))
    print('Photo portal ready:',url,flush=True)
    try:server.serve_forever()
    finally:server.server_close()

if __name__=='__main__':main()
