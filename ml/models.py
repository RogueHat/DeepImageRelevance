import torch
import torch.nn as nn
import torchvision.models as tvmodels

from torch import nn, optim
from torch.nn import functional as F

def create_models():
	sq_feature = tvmodels.mobilenet_v2(pretrained=True)
	sq_feature.classifier = Identity()
	sq_feature.train(False)
	
	vae = VAE()
	
	return sq_feature, vae
	

class Identity(nn.Module):
	def __init__(self):
		super(Identity, self).__init__()

	def forward(self, x):
		return x

class VAE(nn.Module):
	def __init__(self):
		super(VAE, self).__init__()
		
		actives = nn.ReLU
		# ~ drop = nn.Dropout()
		drop = Identity()
		
		self.encoder = nn.Sequential(
			nn.Linear(1280, 1000),
			actives(),
			drop,
			nn.Linear(1000, 600),
			actives(),
			drop
		)
		
		self.mean_layer = nn.Sequential(
			nn.Linear(600, 100),
			actives()
		)	
		self.logvar_layer = nn.Sequential(
			nn.Linear(600, 100),
			actives()
		)
		
		self.decoder = nn.Sequential(
			nn.Linear(100, 600),
			actives(),
			drop,
			nn.Linear(600, 1000),
			actives(),
			drop,
			nn.Linear(1000, 1280)
		)

	def encode(self, x):
		h1 = self.encoder(x)
		return self.mean_layer(h1), self.logvar_layer(h1)

	def reparameterize(self, mu, logvar):
		std = torch.exp(0.5*logvar)
		eps = torch.randn_like(std)
		return mu + eps*std

	def decode(self, z):
		return self.decoder(z)

	def forward(self, x):
		mu, logvar = self.encode(x)
		z = self.reparameterize(mu, logvar)
		return self.decode(z), mu, logvar

	def loss_function(recon_x, x, mu, logvar):
		# ~ BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
		# switched to MSE loss to match range of values from classifier features
		BCE = F.mse_loss(recon_x, x, reduction='sum')

		# see Appendix B from VAE paper:
		# Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
		# https://arxiv.org/abs/1312.6114
		# 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
		KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
		
		return BCE + KLD
