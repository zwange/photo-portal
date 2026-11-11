"""Portal display preferences; no catalog rows or photo files are modified."""
import json,ntpath
from pathlib import Path

CONFIG=Path(__file__).resolve().with_name('config.json')

def _settings():
    try:data=json.loads(CONFIG.read_text(encoding='utf-8-sig'))
    except FileNotFoundError:return {}
    except (OSError,ValueError) as error:raise ValueError(f'Cannot read config.json: {error}') from error
    if not isinstance(data,dict):raise ValueError('config.json must contain an object')
    return data

def catalog_paths():
    settings=_settings()
    db=settings.get('database','data/photo_catalog.sqlite')
    roots=settings.get('photo_roots',['photos'])
    if not isinstance(db,str) or not db.strip():raise ValueError('database must be a path')
    if not isinstance(roots,list) or not roots or any(not isinstance(p,str) or not p.strip() for p in roots):
        raise ValueError('photo_roots must be a nonempty array of paths')
    resolve=lambda value:(CONFIG.parent/Path(value).expanduser()).resolve()
    return resolve(db),list(dict.fromkeys(map(resolve,roots)))

def folder_key(value):return ntpath.normpath(value.replace('/','\\')).rstrip('\\').casefold()

def load_config():
    data=_settings()
    folders=data.get('hidden_folders',[])
    if not isinstance(folders,list) or any(not isinstance(p,str) for p in folders):raise ValueError('hidden_folders must be an array of folder paths')
    result=[]
    for folder in folders:
        folder=folder.strip()
        if not folder:continue
        if not ntpath.isabs(folder):raise ValueError('hidden_folders paths must be absolute')
        result.append(folder_key(folder))
    return dict(hidden_folders=sorted(set(result)))

def is_visible(folder,hidden):
    key=folder_key(folder)
    return not any(key==root or key.startswith(root+'\\') for root in hidden)
