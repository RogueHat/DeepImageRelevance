import sys
sys.path.append('../ml')

import torch
import torchvision.models as tvmodels
import models as mymodels

from torchvision import transforms

class Indexer:	
	def __init__(self, device_name="cpu"):
		self.device_name = device_name
		device = torch.device(device_name)
		
		self.sq_feature, self.vae = mymodels.create_models()
		
		self.vae.load_state_dict(torch.load('../ml/vae.md'))
		self.vae = self.vae.eval()
		
		normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
		self.loader = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor(), normalize])
		
	def get_features(self, im_tensor):
		im_tensor = im_tensor.unsqueeze(0)
		
		im_tensor = self.sq_feature(im_tensor)
		recon_batch, mu, logvar = self.vae(im_tensor)
		
		return mu, logvar
	
	def batch_get_feat(self, tensors):
		im_tensor = torch.stack(tensors)
		im_tensor = self.sq_feature(im_tensor)
		_, mu, logvar = self.vae(im_tensor)
		
		del im_tensor
		return mu, logvar
		
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
