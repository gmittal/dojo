import os
import sys
import time

from absl import app
from absl import flags
from absl import logging
import numpy as np
import torch
import torch.nn as nn
import tensorflow as tf
import tensorflow_datasets as tfds

sys.path.append("{}/..".format(os.path.dirname(os.path.abspath(__file__))))
from utils.train_utils import log_metrics

FLAGS = flags.FLAGS
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

flags.DEFINE_integer('epochs', 10, 'Number of epochs.')
flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate.')
flags.DEFINE_integer('batch_size', 512, 'Batch size.')
flags.DEFINE_integer('latent_dims', 32, 'Latent dimensions.')
flags.DEFINE_integer('log_interval', 10, 'Logging interval.')
flags.DEFINE_boolean('verbose', True, 'Logging toggle.')


class Embedding(nn.Module):
  """MLP-based autoencoder."""

  def __init__(self, feature_size, latent_dims):
    super(Embedding, self).__init__()

    self.encoder = nn.Sequential(*[
        nn.Linear(feature_size, 2048),
        nn.LeakyReLU(),
        # nn.Linear(2048, 2048),
        # nn.LeakyReLU(),
        nn.Linear(2048, latent_dims)
    ])
    self.decoder = nn.Sequential(*[
        nn.Linear(latent_dims, 2048),
        nn.LeakyReLU(),
        # nn.Linear(2048, 2048),
        # nn.LeakyReLU(),
        nn.Linear(2048, feature_size)
    ])

  def encode(self, x):
    return self.encoder(x)

  def decode(self, z):
    return self.decoder(z)

  def forward(self, x):
    z = self.encode(x)
    x_hat = self.decode(z)
    return x_hat


class Classifier(nn.Module):
  """CIFAR-10 classifier for learned embeddings."""

  def __init__(self, embedding_dims, num_classes):
    super(Classifier, self).__init__()

    self.net = nn.Sequential(*[
        nn.Linear(embedding_dims, 1024),
        nn.LeakyReLU(),
        nn.Linear(1024, 512),
        nn.LeakyReLU(),
        nn.Linear(512, num_classes),
        nn.Softmax()
    ])

  def forward(self, x):
    return self.net(x)


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


def mean_squared_error(logits, labels):
  return torch.mean(torch.sum(torch.square(logits - labels)))


def compute_metrics(x_hat, x, mean, logvar):
  mse_loss = mean_squared_error(x_hat, x).mean()
  return {'loss': mse_loss.item()}


def eval(model, eval_ds, epoch):
  model.eval()
  logits, mean, logvar = model(eval_ds)
  return compute_metrics(logits, eval_ds, mean, logvar)


def train_step(optimizer, model, batch):
  """Train the model on a single batch."""

  batch = batch.to(DEVICE)
  optimizer.zero_grad()
  batch_hat = model(batch)

  # Compute loss and backprop
  loss = mean_squared_error(batch_hat, batch)
  loss.backward()
  optimizer.step()

  train_metrics = {
      'loss': loss.item(),
  }

  return optimizer, train_metrics


def train(model, train_ds):
  """Train the model."""

  optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.learning_rate)
  num_batches = tf.data.experimental.cardinality(train_ds)

  for epoch in range(FLAGS.epochs):
    model.train()
    start_time = time.time()

    data = tfds.as_numpy(train_ds)

    for i, batch in enumerate(data):
      images = torch.tensor(batch['image'], dtype=torch.float32)
      labels = torch.tensor(batch['label'], dtype=torch.float32)
      del labels  # unused (for now)

      # flatten and normalize
      images = images.reshape(images.shape[0], -1)
      images = images / 255

      optimizer, train_metrics = train_step(optimizer, model, images)

      if FLAGS.verbose and i % FLAGS.log_interval == 0:
        elapsed = time.time() - start_time
        ms_per_batch = elapsed * 1000 / (FLAGS.log_interval * (i + 1))
        train_metrics['lr'] = FLAGS.learning_rate
        train_metrics['ms/batch'] = ms_per_batch
        log_metrics(train_metrics, i, num_batches, epoch=epoch)

  return optimizer


def main(argv):
  del argv  # unused

  train_ds = tfds.load('cifar10',
                       split='train',
                       batch_size=FLAGS.batch_size,
                       shuffle_files=True)
  test_ds = tfds.load('cifar10',
                      split='test',
                      batch_size=FLAGS.batch_size,
                      shuffle_files=True)

  embedding = Embedding(32 * 32 * 3, FLAGS.latent_dims)
  report_model(embedding)

  train(embedding, train_ds)
  # Save


if __name__ == '__main__':
  app.run(main)
