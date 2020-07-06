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


class Autoencoder(nn.Module):
  """MLP-based autoencoder."""

  def __init__(self, feature_size, latent_dims):
    super(Autoencoder, self).__init__()

    self.encoder = nn.Sequential(*[
        nn.Linear(feature_size, 100),
        nn.LeakyReLU(),
        nn.Linear(100, latent_dims)
    ])
    self.decoder = nn.Sequential(*[
        nn.Linear(latent_dims, 100),
        nn.LeakyReLU(),
        nn.Linear(100, feature_size)
    ])

  def encode(self, x):
    return self.encoder(x)

  def decode(self, z):
    return self.decoder(z)

  def forward(self, x):
    z = self.encode(x)
    y = self.decode(z)
    return y


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


def main(unused_argv):
  del unused_argv
  model = Autoencoder(10, 5)

  report_model(model)


if __name__ == '__main__':
  app.run(main)
