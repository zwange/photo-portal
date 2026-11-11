"""Index local still photos and create JPEG thumbnails for Photo Portal."""
from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3

from PIL import ExifTags, Image, ImageOps
from classify_hierarchy import schema as hierarchy_schema
from portal_config import catalog_paths

CACHE = '.photo-portal-thumbs'
EXTENSIONS = {'.jpg','.jpeg','.png','.webp','.tif','.tiff','.bmp'}
SCHEMA = """
CREATE TABLE IF NOT EXISTS photos (
    id INTEGER PRIMARY KEY, root_folder TEXT, folder_path TEXT, file_name TEXT,
    relative_path TEXT, full_path TEXT NOT NULL UNIQUE, thumbnail_path TEXT,
    category TEXT, category_margin REAL, fgaesq_score REAL, fgaesq_rank INTEGER,
    category_rank INTEGER, folder_rank INTEGER, camera_make TEXT, camera_model TEXT,
    lens_model TEXT, focal_length_mm REAL, aperture_f_number REAL, iso INTEGER,
    exposure_time_description TEXT, date_taken TEXT, width_px INTEGER, height_px INTEGER,
    thumbnail_status TEXT, thumbnail_error TEXT, metadata_status TEXT, exif_json TEXT,
    analysis_status TEXT, artimuse_score REAL, artimuse_status TEXT, artimuse_error TEXT,
    artimuse_scored_at TEXT
);
CREATE TABLE IF NOT EXISTS artimuse_runs (
    run_id TEXT PRIMARY KEY, started_at TEXT, finished_at TEXT,
    status TEXT, model TEXT, revision TEXT, batch_size INTEGER, preprocessing TEXT,
    precision TEXT, eligible INTEGER, completed INTEGER, failed INTEGER, error TEXT
);
CREATE TABLE IF NOT EXISTS visual_hash_runs(run_id TEXT PRIMARY KEY,created_at TEXT,scope TEXT,summary_json TEXT);
CREATE TABLE IF NOT EXISTS photo_visual_hashes(photo_id INTEGER REFERENCES photos(id),version TEXT,
    phash64 TEXT,phash256 TEXT,dhash64 TEXT,decoded_sha256 TEXT,thumbnail_size INTEGER,
    thumbnail_mtime_ns INTEGER,PRIMARY KEY(photo_id,version));
CREATE TABLE IF NOT EXISTS visual_matches(run_id TEXT REFERENCES visual_hash_runs(run_id),
    photo_a INTEGER REFERENCES photos(id),photo_b INTEGER REFERENCES photos(id),decision TEXT,
    metrics_json TEXT,PRIMARY KEY(run_id,photo_a,photo_b));
CREATE TABLE IF NOT EXISTS visual_groups(run_id TEXT REFERENCES visual_hash_runs(run_id),group_id INTEGER,
    representative_photo_id INTEGER REFERENCES photos(id),member_count INTEGER,PRIMARY KEY(run_id,group_id));
CREATE TABLE IF NOT EXISTS visual_group_members(run_id TEXT,group_id INTEGER,photo_id INTEGER REFERENCES photos(id),
    PRIMARY KEY(run_id,photo_id),FOREIGN KEY(run_id,group_id) REFERENCES visual_groups(run_id,group_id));
CREATE INDEX IF NOT EXISTS idx_visual_members_photo ON visual_group_members(photo_id);
"""


def number(value, cast=float):
    try:
        return cast(value)
    except (ValueError, TypeError, OverflowError, ZeroDivisionError):
        return None


def metadata(image):
    try:
        exif = image.getexif()
        values = {ExifTags.TAGS.get(k,str(k)):v for k,v in
                  (dict(exif) | exif.get_ifd(ExifTags.IFD.Exif)).items()}
    except (OSError, ValueError, TypeError):
        values = {}
    raw_date = str(values.get('DateTimeOriginal') or values.get('DateTime') or '')
    date = raw_date[:10].replace(':','-') + raw_date[10:] if raw_date else None
    exposure = values.get('ExposureTime')
    return dict(camera_make=str(values.get('Make') or '').strip() or None,
                camera_model=str(values.get('Model') or '').strip() or None,
                lens_model=str(values.get('LensModel') or '').strip() or None,
                focal_length_mm=number(values.get('FocalLength')),
                aperture_f_number=number(values.get('FNumber')),
                iso=number(values.get('ISOSpeedRatings') or values.get('PhotographicSensitivity'),int),
                exposure_time_description=f'{exposure} s' if exposure is not None else None,
                date_taken=date, metadata_status='ok' if values else 'no_exif',
                exif_json=json.dumps(values,default=str,ensure_ascii=False))


def prepare(root, path):
    rel = path.relative_to(root)
    thumb = root / CACHE / rel.parent / (rel.name + '.jpg')
    row = dict(root_folder=str(root), folder_path=str(path.parent), file_name=path.name,
               relative_path=str(rel),full_path=str(path),thumbnail_path=str(thumb))
    try:
        with Image.open(path) as original:
            row.update(metadata(original))
            oriented = ImageOps.exif_transpose(original)
            row.update(width_px=oriented.width,height_px=oriented.height)
            oriented.thumbnail((1024,1024),Image.Resampling.LANCZOS)
            thumb.parent.mkdir(parents=True,exist_ok=True)
            if not thumb.resolve().is_relative_to(root):
                raise ValueError('Thumbnail path is outside the photo root')
            temp = thumb.with_suffix('.tmp')
            oriented.convert('RGB').save(temp,'JPEG',quality=90)
            temp.replace(thumb)
        # analysis_status is a compatibility field for older catalogs.
        row.update(thumbnail_status='ok',thumbnail_error=None,analysis_status='ok')
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        row.update(thumbnail_status='error',thumbnail_error=str(exc),analysis_status=None)
    return row


def main():
    database, roots = catalog_paths()
    for root in roots:
        if not root.is_dir():
            raise SystemExit(f'Photo root does not exist: {root}')
        if any(other != root and root.is_relative_to(other) for other in roots):
            raise SystemExit('Photo roots must not be nested within one another.')
    database.parent.mkdir(parents=True,exist_ok=True)
    scanned = errors = 0
    with closing(sqlite3.connect(database,timeout=60)) as db,db:
        db.executescript(SCHEMA)
        hierarchy_schema(db)
        known = dict(db.execute("SELECT full_path,thumbnail_path FROM photos WHERE thumbnail_status='ok'"))
        for root in roots:
            for folder, dirs, files in os.walk(root):
                dirs[:] = sorted(d for d in dirs if not d.startswith('.') and d.casefold()!='thumbnails')
                for name in sorted(files):
                    path = Path(folder)/name
                    if path.suffix.lower() not in EXTENSIONS or not path.resolve().is_relative_to(root):
                        continue
                    if (thumbnail := known.get(str(path))) and Path(thumbnail).is_file():
                        continue
                    row = prepare(root,path)
                    columns = ','.join(row)
                    placeholders = ','.join('?' for _ in row)
                    updates = ','.join(f'{key}=excluded.{key}' for key in row if key!='full_path')
                    db.execute(f'INSERT INTO photos({columns}) VALUES ({placeholders}) '
                               f'ON CONFLICT(full_path) DO UPDATE SET {updates}',tuple(row.values()))
                    scanned += 1
                    errors += row['thumbnail_status'] != 'ok'
                    if scanned%100 == 0:
                        db.commit()
                        print(f'Indexed {scanned:,} images ({errors:,} thumbnail errors)',flush=True)
    print(f'Indexed {scanned:,} new or retried images ({errors:,} thumbnail errors) into {database}')


if __name__ == '__main__':
    main()
