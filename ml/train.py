import random

import torch
import torch.nn as nn
import torch.utils.data as torchdata
import torchvision.models as tvmodels
import models as mymodels

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, RandomSampler
from torch.autograd import Variable
from torch.nn import functional as F
from torch import optim
from PIL import Image
from os import path

def get_loss(data, sq_feature, vae, device):
	data = data.to(device)
	
	data = sq_feature(data)
	data = Variable(data, requires_grad=False)
	recon_batch, mu, logvar = vae(data)
	
	loss = mymodels.VAE.loss_function(recon_batch, data, mu, logvar)
	return loss

def repeating_dataloader_generator(dataset, bsize):
	while True:
		dataloader = DataLoader(dataset, batch_size=bsize, shuffle=True)
		for (data, size) in dataloader:
			yield (data, size)

if __name__ == '__main__':
	
	# configs
	loss_size = 10
	bsize = 100
	validate_size = 10000

	# dataset stuff
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
	loader = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor(), normalize])
	traindir = '../danbooru/'
	image_dataset = datasets.ImageFolder(traindir, loader)
	validate_set, training_set = torchdata.random_split(image_dataset, (validate_size, len(image_dataset)- validate_size))
	print(len(validate_set), len(training_set))
	
	training_dataloader = DataLoader(training_set, batch_size=bsize, shuffle=True)
	validate_dataloader = repeating_dataloader_generator(validate_set, bsize)
	
	# torch stuff
	device = torch.device("cpu")

	sq_feature, vae = mymodels.create_models()
	sq_feature = sq_feature.to(device)
	
	# ~ if path.exists('vae.md'):
		# ~ vae.load_state_dict(torch.load('vae.md'))

	vae = vae.to(device)
	optimizer = optim.Adam(vae.parameters(), lr=1e-4)

	loss_list = []
	
	# training
	for i, (training_data, _) in enumerate(training_dataloader):
		
		optimizer.zero_grad()
		training_loss = get_loss(training_data, sq_feature, vae, device)
		training_loss.backward()
		optimizer.step()
		
		(validate_data, _) = next(validate_dataloader)
		validate_loss = get_loss(validate_data, sq_feature, vae, device)
		
		loss_list.append(validate_loss)
		if len(loss_list) > loss_size:
			loss_list = loss_list[-loss_size:]
		
		loss_stack = torch.stack(loss_list)
		std_loss = loss_stack.std()
		
		print(i, validate_loss.item(), std_loss.item())
		
		torch.cuda.empty_cache()
		if i == 1000:
			break
		
		# ~ if len(loss_list) == loss_size and std_loss.item() < 250:
			# ~ break
			
	torch.save(vae.state_dict(), 'vae_new.md')
	print('VAE saved')
