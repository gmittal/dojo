from absl import app
from absl import flags
from absl import logging
import numpy as np
import torch
import torch.nn as nn

FLAGS = flags.FLAGS

flags.DEFINE_string('data', './articles.json', 'JSON file path.')
flags.DEFINE_integer('n_sim', 100, 'Number of embeddings to visualize.')
flags.DEFINE_integer('k', 10, 'Number of nearest neighbors.')


class VAE(nn.Module):
  """MLP-based variational autoencoder."""

  def __init__(self, feature_dims, latent_dims):
    super(VAE, self).__init__()

    self.features = feature_dims
    self.latents = latent_dims

    self.encoder = nn.Sequential(*[
        nn.Linear(self.features, 1024),
        nn.LeakyReLU(),
        nn.Linear(1024, 256),
        nn.LeakyReLU(),
        nn.Linear(256, 64),
        nn.LeakyReLU()
    ])

    self.mu = nn.Linear(64, self.latents)
    self.logvar = nn.Linear(64, self.latents)

    self.decoder = nn.Sequential(*[
        nn.Linear(self.latents, 64),
        nn.LeakyReLU(),
        nn.Linear(64, 256),
        nn.LeakyReLU(),
        nn.Linear(256, 1024),
        nn.LeakyReLU(),
        nn.Linear(1024, self.features)
    ])

  def encode(self, x):
    x = self.encoder(x)
    mu = self.mu(x)
    logvar = self.logvar(x)
    return mu, logvar

  def reparameterize(self, mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

  def decode(self, z):
    return self.decoder(z)

  def forward(self, x):
    mu, logvar = self.encode(x)
    z = self.reparameterize(mu, logvar)
    return self.decode(z), mu, logvar


class Unflatten(nn.Module):

  def __init__(self, channels=32, dims=4):
    super(Unflatten, self).__init__()

    self.channels = channels
    self.dims = dims

  def forward(self, x):
    return x.view(x.size(0), self.channels, self.dims, self.dims)


class ConvVAE(nn.Module):
  """CNN-based variational autoencoder."""

  def __init__(self, feature_dims, latent_dims):
    super(ConvVAE, self).__init__()

    self.features = feature_dims
    self.latents = latent_dims

    self.encoder = nn.Sequential(*[
        nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
        nn.Flatten(),
        nn.Linear(64 * (self.features // 4)**2, 512),
        nn.LeakyReLU(),
        nn.Linear(512, 256)
    ])

    self.mu = nn.Linear(256, self.latents)
    self.logvar = nn.Linear(256, self.latents)

    self.decoder = nn.Sequential(*[
        nn.Linear(self.latents, 256),
        nn.LeakyReLU(),
        nn.Linear(256, 512),
        nn.LeakyReLU(),
        nn.Linear(512, 64 * (self.features // 4)**2),
        Unflatten(channels=64, dims=self.features // 4),
        nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(),
        nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)
    ])

  def encode(self, x):
    x = self.encoder(x)
    mu = self.mu(x)
    logvar = self.logvar(x)
    return mu, logvar

  def reparameterize(self, mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

  def decode(self, z):
    return self.decoder(z)

  def forward(self, x):
    assert x.shape[-1] % 4 == 0, 'Invalid max_len. Must be divisible by 4.'

    x = x.unsqueeze(1)
    mu, logvar = self.encode(x)
    z = self.reparameterize(mu, logvar)
    reconstruction = self.decode(z).squeeze(1)
    return reconstruction, mu, logvar


def report_model(model):
  """Report model architecture and size."""
  ps = []
  for _, p in model.named_parameters():
    ps.append(np.prod(p.size()))
  num_params = sum(ps)
  mb = num_params * 4 / 1024 / 1024
  logging.info('Model parameters: {} (~= {:.1f}MB)'.format(num_params, mb))
  logging.info(model)
  return mb


def main(argv):
  del argv

  model = ConvVAE(28, 5)
  
  report_model(ConvVAE(10, 5))


if __name__ == '__main__':
  app.run(main)
