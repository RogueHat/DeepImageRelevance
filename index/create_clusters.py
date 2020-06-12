from io import BytesIO
from multiprocessing import Pool
from datetime import datetime
from kmeans_pytorch import kmeans

import sqlite3, torch, shelve, random

def deserialize(tup):
	(dbid, tensorbuf) = tup
	with BytesIO(tensorbuf) as tensorbyte:
		tensor = torch.load(tensorbyte)[0].unsqueeze(0)
	tensor = tensor.detach()
	return (dbid, tensor)

if __name__ == '__main__':
	dbid_lst = []
	tensor_lst = []
	bsize = 50000
	
	dbid_select = 'SELECT dbid FROM features'
	temp_table = 'CREATE TEMPORARY TABLE select_dbids (dbid TEXT PRIMARY KEY)'
	temp_insert = 'INSERT INTO select_dbids VALUES(?)'
	select_stmt = '''
		SELECT fts.dbid, fts.tensor 
		FROM features as fts
		INNER JOIN select_dbids
		ON select_dbids.dbid = fts.dbid
	'''
	
	with sqlite3.connect('deeprelevance.db', timeout=60*5) as conn:
		c = conn.cursor()
		c.execute(dbid_select)
		
		dbid_tups = c.fetchall()
		dbid_tups = random.sample(dbid_tups, int(len(dbid_tups)*0.10))
		print('Selected random dbids')
		
		c.execute(temp_table)
		c.executemany(temp_insert, dbid_tups)
		c.execute(select_stmt)
		
		def batch_enumerate(bs):
			batch = c.fetchmany(bs)
			while(len(batch)):
				yield batch
				batch = c.fetchmany(bs)
		
		batches = batch_enumerate(bsize)
		
		for i, batch in enumerate(batches):
			sub_lst = list(map(deserialize, batch))
			sub_dbids, sub_tensors = zip(*sub_lst)
			dbid_lst.extend(sub_dbids)
			tensor_lst.extend(sub_tensors)
			
			print('Batch {} completed. Processed {}. {}'.format(i+1, len(dbid_lst), datetime.now()))
	
	x = torch.stack(tensor_lst)
	cluster_ids, cluster_centers = kmeans(X=x, num_clusters=100, distance='cosine', device=torch.device('cpu'))
	
	torch.save(cluster_centers, 'centroids.tensor')
		
