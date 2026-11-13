"""Check the portable catalog and gallery using generated images only."""
from contextlib import closing
import gc
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import catalog
import portal_config
import score_para_catalog
import server


class SmokeTest(unittest.TestCase):
    def test_index_publish_and_gallery(self):
        with tempfile.TemporaryDirectory(prefix='.test-data-',dir=Path(__file__).resolve().parents[1]) as tmp:
            base=Path(tmp)
            config=base/'config.json'
            config.write_text(json.dumps({'database':'catalog.sqlite','photo_roots':['photos']}),encoding='utf-8')
            with patch.object(portal_config,'CONFIG',config):
                database,roots=portal_config.catalog_paths()
                files=('Airshow/Day 1/one.jpg','Airshow/Day 2/two.jpg','Airfield/three.jpg')
                for i,name in enumerate(files):
                    path=roots[0]/name
                    path.parent.mkdir(parents=True,exist_ok=True)
                    Image.new('RGB',(1200,800),(60*i,80,120)).save(path)
                catalog.main()
                catalog.main()
                staged=base/'staging.sqlite'
                with closing(sqlite3.connect(database)) as db,db:
                    rows=db.execute('SELECT id,thumbnail_path FROM photos ORDER BY id').fetchall()
                    self.assertEqual(len(rows),3)
                    for ident,thumb in rows:
                        with Image.open(thumb) as im:self.assertEqual(im.width,1024)
                        db.execute("""INSERT INTO photo_classifications
                            (photo_id,primary_category,subcategory,subject_tags_json,scene_tags_json,
                            description,animal_present,model,policy,status)
                            VALUES (?,'vehicles','aircraft','["airplane"]','[]','Test aircraft',0,'test','test','ok')""",(ident,))
                with closing(sqlite3.connect(staged)) as db,db:
                    db.execute('CREATE TABLE scores(id INTEGER PRIMARY KEY,thumbnail_path TEXT,score REAL,scored_at TEXT)')
                    db.executemany("INSERT INTO scores VALUES (?,?,?,'test')",[(i,p,35+i) for i,p in rows])
                with patch.object(score_para_catalog,'DB',database):
                    score_para_catalog.publish(dict(run_id='test',started_at='test',model='ArtiMuse_PARA',revision='test'),staged)
                with patch.multiple(server,DB=database,ROOTS=roots,
                                    ROOT_KEYS=tuple(map(lambda p:portal_config.folder_key(str(p)),roots)),
                                    APP=base,TAG_CACHE={'expires':0}):
                    result=server.gallery(dict(categories=['vehicles'],camera_only=False,deduplicate=False,
                                               folder_cap=1,per_category=10))
                    self.assertEqual([p['artimuse_score'] for p in result['photos']],[38,36])
                    self.assertEqual(result['photos'][0]['artimuse_percentile'],100)
                    self.assertIn('aircraft',server.metadata()['taxonomy']['vehicles'])
                    query=server.custom_query(dict(sql="SELECT artimuse_percentile(artimuse_score) FROM photos ORDER BY id"))
                    self.assertEqual(query['rows'],[[100/3],[200/3],[100]])
                    self.assertIsNone(server.visual_catalog()['method'])
                    with closing(sqlite3.connect(database)) as db,db:
                        db.executescript("""
                            CREATE TABLE sscd_runs(run_id TEXT,cutoff REAL,status TEXT,finished_at TEXT);
                            CREATE TABLE sscd_group_members(run_id TEXT,group_id INTEGER,photo_id INTEGER);
                            INSERT INTO sscd_runs VALUES ('done',0.80,'complete','2026-01-01'),
                                                         ('newer',0.90,'running',NULL);
                            INSERT INTO sscd_group_members VALUES ('done',1,2),('done',1,3);
                        """)
                    groups=server.metadata()['visual']
                    self.assertEqual((groups['method'],groups['run_id'],groups['cutoff']),('sscd','done',.8))
                    self.assertEqual([p['id'] for p in server.versions(2)['photos']],[3,2])
                    selected=server.gallery(dict(categories=['vehicles'],camera_only=False,deduplicate=False,
                                                 folder_cap=0,per_category=10))
                    self.assertEqual([p['id'] for p in selected['photos']],[3,1])
            gc.collect()  # Release SQLite cursors before deleting the Windows fixture.


if __name__=='__main__':
    unittest.main()
