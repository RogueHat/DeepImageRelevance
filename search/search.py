import sys
sys.path.append('../index')

from indexer import Indexer

from io import BytesIO
from PIL import Image
from multiprocessing import Pool
from scipy.stats import norm as spnorm

import torch
import torch.nn as nn

import numpy as np
import sqlite3, time, dhash

def deserialize(tup):
	import torch as dstorch
	(dbid, tensorbuf) = tup
	with BytesIO(tensorbuf) as tensorbyte:
		tensor = dstorch.load(tensorbyte)
	mu, var = tensor[0], tensor[1]
	mu, var = mu.unsqueeze(0), var.unsqueeze(0)
	return (dbid, mu, var)

def preprocess(im):
	thumbsize = (512, 512)
	small = im.copy()
	small.thumbnail(thumbsize)
	smallsize = small.size
	centerloc = tuple(int((a-b)/2) for a,b in zip(thumbsize, smallsize))
	square = Image.new('RGB', thumbsize)
	square.paste(small, centerloc)
	return square

def integer_hamming(x, y):
	return bin(x^y).count('1')

class Search:
	def __init__(self, cluster_size=2000, dhash_size=4):
		self.cluster_size = cluster_size
		self.dhash_size = dhash_size
		self.indexer = Indexer()
		
		self.db_path = '../index/deeprelevance.db'
		self.temp_table = 'CREATE TEMPORARY TABLE dhash_filtered (dbid TEXT PRIMARY KEY)'
		self.temp_insert = 'INSERT INTO dhash_filtered VALUES(?)'
		
		self.dhash_stmt = 'SELECT dbid, dhash FROM features'
		self.feats_stmt = '''
			SELECT fts.dbid, fts.tensor
			FROM features as fts
			INNER JOIN dhash_filtered
			ON fts.dbid = dhash_filtered.dbid
		'''
	
	def get_danbooru_ids(self, im, result_size=20):
		assert result_size <= self.cluster_size
		p = Pool()
		
		old_time = time.time()
		
		im = preprocess(im)
		targethash = dhash.dhash_int(im, self.dhash_size)
		im_tensor = self.indexer.to_tensor(im)
		xtarget, _ = self.indexer.get_features(im_tensor)
		
		sub_time = time.time()
		with sqlite3.connect(self.db_path, timeout=60*5) as conn:
			c = conn.cursor()
			c.execute(self.dhash_stmt)
			tups = c.fetchall()
			
		tups = list(sorted(tups, key = lambda tup: integer_hamming(targethash, tup[1])))[:self.cluster_size]
		fnames = [(tup[0],) for tup in tups]
		print('dhash filter', time.time()-sub_time)

		sub_time = time.time()
		with sqlite3.connect(self.db_path, timeout=60*5) as conn:
			c = conn.cursor()
			c.execute(self.temp_table)
			c.executemany(self.temp_insert, fnames)
			c.execute(self.feats_stmt)
			tups = c.fetchall()
		print('feature get', time.time()-sub_time)

		sub_time = time.time()
		tups = list(map(deserialize, tups))
		print('feature deserialize', time.time()-sub_time)

		sub_time = time.time()
		mu = torch.stack([tup[1] for tup in tups])
		logvar = torch.stack([tup[2] for tup in tups])
		x = torch.stack([xtarget for i in range(len(tups))])
		
		cos = nn.CosineSimilarity(dim=-1)
		similarity_scores = cos(mu, x).flatten()
		ordered_scores, idxs = torch.topk(similarity_scores, result_size, largest=True, sorted=True)
		ordered_scores = ordered_scores.detach().numpy()
		ordered_scores = ordered_scores * ordered_scores

		dbids = [tups[i][0] for i in idxs]
		dbid_pval_pairs = list(zip(dbids, ordered_scores))
		print('feature topk', time.time()-sub_time)
		
		run_time = time.time() - old_time
		return dbid_pval_pairs, run_time
