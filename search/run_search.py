from search import Search
from PIL import Image

searcher = Search(10000)
db_path = searcher.db_path

with Image.open('test_img.jpg') as im:
	dbid_pval_pairs, run_time = searcher.get_danbooru_ids(im)

dbids = [tup[0] for tup in dbid_pval_pairs]
pvals = [tup[1] for tup in dbid_pval_pairs]

import sqlite3
temp_table = 'CREATE TEMPORARY TABLE selected_dbids (dbid TEXT PRIMARY KEY)'
temp_insert = 'INSERT INTO selected_dbids VALUES(?)'
path_stmt = '''
	SELECT fpaths.dbid, fpaths.filepath
	FROM filepaths AS fpaths
	INNER JOIN selected_dbids
	ON fpaths.dbid = selected_dbids.dbid
'''
dbid_tups = [(dbid, ) for dbid in dbids]

with sqlite3.connect(db_path) as conn:
	c = conn.cursor()
	
	c.execute(temp_table)
	c.executemany(temp_insert, dbid_tups)
	
	c.execute(path_stmt)
	dbid_fpath_pairs = c.fetchall()

dbid_fpath_dct = {dbid:fpath for dbid, fpath in dbid_fpath_pairs}
ordered_fpaths = [dbid_fpath_dct[dbid] for dbid in dbids]
entries = zip(pvals, ordered_fpaths)

import shutil, os
shutil.rmtree('results')
os.makedirs('results')

fpath_template = 'results/{}_{}.jpg'
for i, (pval, fpath) in enumerate(entries):
	new_fpath = fpath_template.format(i, round(pval,4))
	shutil.copy(fpath, new_fpath)
	
