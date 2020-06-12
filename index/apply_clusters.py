from io import BytesIO
from multiprocessing import Pool
from datetime import datetime
from kmeans_pytorch import kmeans
from indexer import Indexer

import sqlite3, torch, shelve, random

indexer = Indexer()

def deserialize(tup):
	(dbid, tensorbuf) = tup
	with BytesIO(tensorbuf) as tensorbyte:
		tensor = torch.load(tensorbyte)[0].unsqueeze(0)
	tensor = tensor.detach()
	return (dbid, tensor)

def worker(tup):
	dbid, tensor = tup
	cid = indexer.get_cluster_id(tensor)
	return cid, dbid

if __name__ == '__main__':
	bsize = 50000
	dbid_tensor = 'SELECT dbid, tensor FROM features'
	dbid_update = 'UPDATE features SET clusterid=? WHERE dbid=?'
	
	with sqlite3.connect('deeprelevance.db', timeout=60*5) as conn:
		reader = conn.cursor()
		writer = conn.cursor()
		
		reader.execute(dbid_tensor)
		def batch_gen():
			batch = reader.fetchmany(bsize)
			while len(batch):
				yield batch
				batch = reader.fetchmany(bsize)
		batches = batch_gen()
		
		for i, batch in enumerate(batches):
			tensortups = list(map(deserialize, batch))
			cidtups = list(map(worker, tensortups))
			
			writer.executemany(dbid_update, cidtups)
			conn.commit()
			print('Processed Batch {}'.format(i))
			
	
	
	
