from absl import app
from absl import flags
from absl import logging
import numpy as np
import torch
import torch.nn as nn

FLAGS = flags.FLAGS

flags.DEFINE_enum('format', True)
