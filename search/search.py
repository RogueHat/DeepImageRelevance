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
	(dbid, tensorbuf) = tup
	with BytesIO(tensorbuf) as tensorbyte:
		tensor = torch.load(tensorbyte).detach()
	mu, var = tensor[0], tensor[1]
	mu, var = mu.unsqueeze(0), var.unsqueeze(0)
	return (dbid, mu, var)

def integer_hamming(x, y):
	return bin(x^y).count('1')
	
def hamming_worker(targethash):
	def inner_worker(tup):
		dbid, otherhash = tup
		return dbid, integer_hamming(targethash, otherhash)
	return inner_worker

class Search:
	def __init__(self, cluster_size=2000, dhash_size=4):
		self.cluster_size = cluster_size
		self.dhash_size = dhash_size
		self.indexer = Indexer()
		
		self.db_path = '../index/deeprelevance.db'
		self.temp_table = 'CREATE TEMPORARY TABLE dhash_filtered (dbid TEXT PRIMARY KEY)'
		self.temp_insert = 'INSERT INTO dhash_filtered VALUES(?)'
		
		self.dhash_stmt = 'SELECT dbid, dhash FROM features'
		self.dhash_bit_stmt = 'SELECT dbid, (dhash|?)&~(dhash&?) FROM features'
		
		self.feats_stmt = '''
			SELECT fts.dbid, fts.tensor
			FROM features as fts
			INNER JOIN dhash_filtered
			ON dhash_filtered.dbid = fts.dbid
		'''
	
	def get_dbids_kmeans(self, im, result_size=20, dist_type='cosine'):
		
		# ~ temp_table = 'CREATE TEMPORARY TABLE clusters (dbid TEXT PRIMARY KEY)'
		# ~ temp_insert_cluster = '''
			# ~ INSERT INTO clusters 
			# ~ SELECT dbid 
			# ~ FROM features 
			# ~ WHERE clusterid=?
		# ~ '''
		# ~ tensor_select = '''
			# ~ SELECT features.dbid, features.tensor 
			# ~ FROM features
			# ~ INNER JOIN clusters
			# ~ ON clusters.dbid = features.dbid
		# ~ '''
		
		one_access_stmt = 'SELECT dbid, tensor FROM features WHERE clusterid=?'
		
		old_time = time.time()
		
		im = self.indexer.preprocess(im)
		im_tensor = self.indexer.to_tensor(im)
		xtarget, _ = self.indexer.get_features(im_tensor)
		cid = self.indexer.get_cluster_id(xtarget)
		
		sub_time = time.time()
		with sqlite3.connect(self.db_path, timeout=60*5) as conn:
			c = conn.cursor()
			c.execute(one_access_stmt, (cid, ))
			tups = c.fetchall()
		print('feature get. Cluster size: {} {}'.format(len(tups), time.time()-sub_time) )
			
		sub_time = time.time()
		tups = list(map(deserialize, tups))
		print('feature deserialize', time.time()-sub_time)
		
		sub_time = time.time()
		dbids, mulst, logvarlst = list(zip(*tups))
		mu, logvar = torch.stack(mulst), torch.stack(logvarlst)
		x = torch.stack([xtarget for i in range(len(tups))])
		
		if dist_type == 'z_score':
			z_scores = Indexer.get_zscore(x, mu, logvar)
			z_scores = torch.norm(z_scores, dim=(1,2))
			ordered_z_scores, idxs = torch.topk(z_scores, result_size, largest=False, sorted=True)
			ordered_z_scores = ordered_z_scores.detach().numpy()
			ordered_scores = spnorm.sf(abs(ordered_z_scores)) * 2
			ordered_scores = [score.item() for score in ordered_scores]
		else:
			cos = nn.CosineSimilarity(dim=-1)
			similarity_scores = cos(mu, x).flatten()
			ordered_scores, idxs = torch.topk(similarity_scores, result_size, largest=True, sorted=True)
			ordered_scores = ordered_scores.detach().numpy()
			ordered_scores = ordered_scores * ordered_scores
			ordered_scores = [score.item() for score in ordered_scores]

		select_dbids = [dbids[i] for i in idxs]
		dbid_pval_pairs = list(zip(select_dbids, ordered_scores))
		print('feature topk', time.time()-sub_time)
		
		run_time = time.time() - old_time
		return dbid_pval_pairs, dbid_pval_pairs, run_time
		
				
	def get_danbooru_ids(self, im, result_size=20):
		assert result_size <= self.cluster_size
		
		old_time = time.time()
		
		im = self.indexer.preprocess(im)
		targethash = dhash.dhash_int(im, self.dhash_size)
		im_tensor = self.indexer.to_tensor(im)
		xtarget, _ = self.indexer.get_features(im_tensor)
		
		sub_time = time.time()
		with sqlite3.connect(self.db_path, timeout=60*5) as conn:
			c = conn.cursor()
			# ~ c.execute(self.dhash_stmt)
			c.execute(self.dhash_bit_stmt, (targethash, targethash))
			tups = c.fetchall()
		
		# ~ tups = list(map(hamming_worker(targethash), tups))
		tups = [(t[0], bin(t[1]).count('1')) for t in tups]
		tups = list(sorted(tups, key = lambda tup: tup[1]))[:self.cluster_size]
		dhash_tups = tups[:result_size]
		dhash_tups = [(t[0], 1.0 - t[1]/(2*self.dhash_size)) for t in dhash_tups]
		
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
		
		dbids, mulst, logvarlst = list(zip(*tups))
		mu, logvar = torch.stack(mulst), torch.stack(logvarlst)
		x = torch.stack([xtarget for i in range(len(tups))])
		
		# cosine similarity
		cos = nn.CosineSimilarity(dim=-1)
		similarity_scores = cos(mu, x).flatten()
		ordered_scores, idxs = torch.topk(similarity_scores, result_size, largest=True, sorted=True)
		ordered_scores = ordered_scores.detach().numpy()
		ordered_scores = ordered_scores * ordered_scores
		ordered_scores = [score.item() for score in ordered_scores]
		
		# z-score
		# ~ z_scores = Indexer.get_zscore(x, mu, logvar)
		# ~ z_scores = torch.norm(z_scores, dim=(1,2))
		# ~ ordered_z_scores, idxs = torch.topk(z_scores, result_size, largest=False, sorted=True)
		# ~ ordered_z_scores = ordered_z_scores.detach().numpy()
		# ~ ordered_scores = spnorm.sf(abs(ordered_z_scores)) * 2
		# ~ ordered_scores = [score.item() for score in ordered_scores]
		

		select_dbids = [dbids[i] for i in idxs]
		dbid_pval_pairs = list(zip(select_dbids, ordered_scores))
		print('feature topk', time.time()-sub_time)
		
		run_time = time.time() - old_time
		return dbid_pval_pairs, dhash_tups, run_time
