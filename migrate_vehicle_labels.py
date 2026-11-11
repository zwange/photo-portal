"""Promote existing vehicle labels using subject tags; no image inference."""
import argparse
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sqlite3

ROOT = Path(__file__).resolve().parent
from portal_config import catalog_paths
DB,_ = catalog_paths()
TYPES = {
    'aircraft': r'\b(?:airplanes?|aeroplanes?|planes?|aircraft|jets?|helicopters?|biplanes?|warplanes?|airliners?|drones?|fighter jets?|gliders?)\b',
    'road_vehicles': r'\b(?:cars?|trucks?|buses|bus|motorcycles?|bicycles?|suvs?|vans?|taxis?|tractors?|limousines?|pickup trucks?)\b',
    'rail': r'\b(?:trains?|locomotives?|railcars?|trams?|streetcars?|subway trains?)\b',
    'watercraft': r'\b(?:boats?|ships?|ferries|ferry|canoes?|kayaks?|yachts?|sailboats?|watercraft|jet skis?|submarines?)\b',
}


def vehicle_type(raw_tags, description):
    # Prefer subject tags. Use the description when the tags have no specific type.
    tags = json.loads(raw_tags or '[]')
    for value in (' '.join(tags), description or ''):
        value = re.sub(r'\bjet[- ]skis?\b', 'watercraft', value.casefold())
        value = re.sub(r'\b(?:railroad|railway|train) cars?\b', 'railcar', value)
        matches = [kind for kind, pattern in TYPES.items() if re.search(pattern, value)]
        if len(matches) == 1:
            return matches[0]
        if matches:
            break
    return 'other_vehicles'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    uri = DB.as_uri() + ('?mode=rw' if args.apply else '?mode=ro')
    with closing(sqlite3.connect(uri, uri=True, timeout=30)) as db, db:
        if args.apply:
            db.execute('BEGIN IMMEDIATE')
        rows = db.execute("""SELECT photo_id,subject_tags_json,description FROM photo_classifications
            WHERE status='ok' AND primary_category='other' AND subcategory='vehicles' ORDER BY photo_id""").fetchall()
        changes = [(vehicle_type(tags, desc), ident) for ident, tags, desc in rows]
        print(f'{len(changes):,} photos:', dict(Counter(sub for sub, _ in changes)))
        if args.apply and changes:
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            backup = ROOT / f'vehicle_category_migration_{stamp}.json'
            backup.write_text(json.dumps(dict(before=['other','vehicles'],
                after=[dict(photo_id=ident,primary_category='vehicles',subcategory=sub) for sub,ident in changes]),
                indent=2), encoding='utf-8')
            updated = db.executemany("""UPDATE photo_classifications SET primary_category='vehicles',subcategory=?
                WHERE photo_id=? AND primary_category='other' AND subcategory='vehicles' AND status='ok'""", changes)
            if updated.rowcount != len(changes):
                raise RuntimeError('Vehicle labels changed during migration')
            print('Rollback manifest:', backup)


if __name__ == '__main__':
    main()
