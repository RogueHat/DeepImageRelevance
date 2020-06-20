import sys
sys.path.append('../ml')

import torch
import torch.nn as nn
import torchvision.models as tvmodels
import models as mymodels

from torchvision import transforms
from PIL import Image

class Indexer:	
	def __init__(self, device_name="cpu"):
		self.device_name = device_name
		device = torch.device(device_name)
		
		self.sq_feature, self.vae = mymodels.create_models()
		
		self.vae.load_state_dict(torch.load('../ml/vae.md'))
		self.vae = self.vae.eval()
		
		normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
		self.loader = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor(), normalize])
		
		self.centers = torch.load('../index/centroids.tensor')
		self.cos = nn.CosineSimilarity(dim=-1)
	
	def get_cluster_id(self, feat_tensor):
		assert feat_tensor.shape == (1,100)
		x = torch.stack([feat_tensor]*len(self.centers))
		similarity_scores = self.cos(x, self.centers).flatten()
		_, idxs = torch.topk(similarity_scores, 1, largest=True)
		idx = idxs[0].item()
		
		assert idx < len(self.centers)
		return idx
		
	def get_features(self, im_tensor):
		im_tensor = im_tensor.unsqueeze(0)
		
		im_tensor = self.sq_feature(im_tensor)
		recon_batch, mu, logvar = self.vae(im_tensor)
		
		return mu.detach(), logvar.detach()
	
	def batch_get_feat(self, tensors):
		im_tensor = torch.stack(tensors)
		im_tensor = self.sq_feature(im_tensor)
		_, mu, logvar = self.vae(im_tensor)
		
		del im_tensor
		return mu.detach(), logvar.detach()
		
	def batch_get_feat_stacked(self, tensors):
		mu, logvar = self.batch_get_feat(tensors)
		pairs = [torch.stack(tup) for tup in zip(mu, logvar)]
		del mu, logvar
		return pairs
	
	def to_tensor(self, image):
		return self.loader(image).float()
	
	def get_zscore(x, mu, logvar):
		std = torch.exp(0.5*logvar)
		z = (x - mu) / std
		return z
	
	def preprocess(self, im):
		thumbsize = (512, 512)
		small = im.copy()
		small.thumbnail(thumbsize)
		smallsize = small.size
		centerloc = tuple(int((a-b)/2) for a,b in zip(thumbsize, smallsize))
		square = Image.new('RGB', thumbsize)
		square.paste(small, centerloc)
		
		small.close()
		im.close()
		
		return square

if __name__ == '__main__':
	from multiprocessing import Pool
	
	import random, time
	
	p = Pool()
	indexer = Indexer()
	
	tensors = torch.load('centroids.tensor')
	idx = random.randint(0,len(tensors))
	tensor = tensors[idx]
	
	
	start_time = time.time()
	cids = [indexer.get_cluster_id(tensor) for i in range(50000)]
	run_time = time.time() - start_time
	
	cid = cids[0]
	print(idx, cid, run_time)
	print(type(idx), type(cid))
	
