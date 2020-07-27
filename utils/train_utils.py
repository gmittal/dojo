from absl import logging


def log_metrics(metrics, step, total_steps, epoch=None, summary_writer=None):
  """Log metrics.
  
  Args:
    metrics: A dictionary of scalar metrics.
    step: The current step.
    total_steps: The total number of steps.
    epoch: The current epoch.
    summary_writer: A TensorBoard summary writer.
  """
  metrics_str = ''
  for metric in metrics:
    value = metrics[metric]
    if metric == 'lr':
      metrics_str += '{} {:5.4f} | '.format(metric, value)
    else:
      metrics_str += '{} {:5.2f} | '.format(metric, value)

    if summary_writer is not None:
      summary_writer.scalar(metric, value, step)

  if epoch is not None:
    epoch_str = '| epoch {:3d} '.format(epoch)
  else:
    epoch_str = ''

  logging.info('{}| {:5d}/{:5d} steps | {}'.format(epoch_str, step, total_steps,
                                                   metrics_str))