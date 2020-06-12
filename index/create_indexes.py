from datetime import datetime

indexer_start_time = datetime.now()

import os

def batch(iterable, n=5):
	l = list(next(iterable) for i in range(n))
	while len(l):
		yield l
		l = list(next(iterable) for i in range(n))

def getdanboorupairs(fpath):
	tups = os.walk(fpath, followlinks=True)
	for root, dirs, files in tups:
		if len(files):
			for fname in files:
				fpath = os.path.join(root, fname)
				yield fname.split('.')[0], fpath

import sqlite3
create_feature_table_stmt = 'CREATE TABLE IF NOT EXISTS features (dbid TEXT PRIMARY KEY, dhash INTEGER, tensor BLOB, lastupdate TEXT, clusterid INTEGER)'
create_filepath_table_stmt = 'CREATE TABLE IF NOT EXISTS filepaths (dbid TEXT PRIMARY KEY, filepath TEXT)'

with sqlite3.connect('deeprelevance.db', timeout=60*5) as conn:
	c = conn.cursor()
	c.execute(create_feature_table_stmt)
	c.execute(create_filepath_table_stmt)

from indexer import Indexer
from PIL import Image
from io import BytesIO
import dhash, time, torch

def preprocess(im):
	thumbsize = (512, 512)
	small = im.copy()
	small.thumbnail(thumbsize)
	smallsize = small.size
	centerloc = tuple(int((a-b)/2) for a,b in zip(thumbsize, smallsize))
	square = Image.new('RGB', thumbsize)
	square.paste(small, centerloc)
	return square

def get_tensor(indexer):
	def inner_get_tensor(fpath):	
		with Image.open(fpath) as im:
			im = im.convert('RGB')
			im = preprocess(im)
			im_tensor = indexer.to_tensor(im)
		return im_tensor
	return inner_get_tensor

def get_dhash(fpath):
	with Image.open(fpath) as im:
		im = im.convert('RGB')
		im = preprocess(im)
		return dhash.dhash_int(im, 4)
	
def serialize(tensor):
	tensor = tensor.detach()
	bytesio = BytesIO()
	torch.save(tensor, bytesio)
	buf = bytesio.getvalue()
	return buf

def worker(dbid_fpath_pairs):
	start_time = time.time()
	
	dbids = [tup[0] for tup in dbid_fpath_pairs]
	fpaths = [tup[1] for tup in dbid_fpath_pairs]
	
	dhashes = list(map(get_dhash, fpaths))
	
	indexer = Indexer()
	im_tensors = list(map(get_tensor(indexer), fpaths))
	muvar_pairs = indexer.batch_get_feat_stacked(im_tensors)
	cids = list(map(lambda t: indexer.get_cluster_id(t[0]), muvar_pairs))
	muvar_bufs = list(map(serialize, muvar_pairs))
	lastupdates = [indexer_start_time] * len(im_tensors)
	
	feature_rows = list(zip(dbids, dhashes, muvar_bufs, lastupdates, cids))
	
	insert_fpaths_stmt = 'INSERT OR REPLACE INTO filepaths VALUES (?,?)'
	insert_feats_stmt = 'INSERT OR REPLACE INTO features VALUES (?,?,?,?)'
	with sqlite3.connect('deeprelevance.db', timeout=60*5) as conn:
		c = conn.cursor()
		c.executemany(insert_fpaths_stmt, dbid_fpath_pairs)
		c.executemany(insert_feats_stmt, feature_rows)
		conn.commit()
		
	run_time = time.time() - start_time
	print("{} Runtime: {}".format(indexer_start_time, run_time))


from multiprocessing import Pool
if __name__ == '__main__':
	bsize = 12
	
	dbpairs_iterator = getdanboorupairs('../danbooru')
	dbpair_batches = list(batch(dbpairs_iterator, bsize))
	
	p = Pool(5)
	list(p.map(worker, dbpair_batches))
	
	
	
	
	
	
	
	
	
	
	
	 
	
