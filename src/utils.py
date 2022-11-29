import json
import numpy as np
import os
import shutil
import tensorflow as tf
from types import SimpleNamespace
# from .models import get_apply_fn_test, get_model
# from .train_state import get_train_state


########################################################################################################################
#  Args
########################################################################################################################


def log_and_save_args(args):
  print('train args:')
  print_args(args)
  save_args(args, args.save_dir, verbose=True)


def save_args(args, save_dir, verbose=True):
  save_path = save_dir + '/args.json'
  with open(save_path, 'w') as f: json.dump(vars(args), f, indent=4)
  if verbose: print(f'Save args to {save_path}')
  return save_dir + '/args.json'


def load_args(load_dir, verbose=True):
  load_path = load_dir + '/args.json'
  with open(load_path, 'r') as f: args = SimpleNamespace(**json.load(f))
  if verbose: print(f'Load args from {load_path}')
  return args

def print_args(args):
  print(json.dumps(vars(args), indent=4))


########################################################################################################################
#  File Management
########################################################################################################################

def make_dir(path):
  if os.path.exists(path):
    shutil.rmtree(path)
  os.makedirs(path)


def make_dirs(args):
  make_dir(args.save_dir)
  make_dir(args.save_dir + '/ckpts')


########################################################################################################################
#  Seed
########################################################################################################################

def set_global_seed(seed=0):
  np.random.seed(seed)
  tf.random.set_seed(seed)

########################################################################################################################
#  print
########################################################################################################################

def print_stats(t, T, t_incr, t_tot, train_metric, test_metric, init=False):
  prog = t / T * 100
  print(f'{prog:6.2f}% | time: {t_incr:5.1f}s ({t_tot/60:5.1f}m) | step: {t:6d} |',
          f"train acc: {train_metric['accuracy']:.3f}({train_metric['acc'][0]:.3f}, {train_metric['acc'][1]:.3f}) | train ar gap: {abs(train_metric['ar'][0]-train_metric['ar'][1]):.3f} | train loss: {train_metric['loss']:.3f} |test acc: {test_metric['accuracy']:.3f} ({test_metric['acc'][0]:.3f}, {test_metric['acc'][1]:.3f}) | test ar gap: {abs(test_metric['ar'][0] - test_metric['ar'][1]):.4f}")
