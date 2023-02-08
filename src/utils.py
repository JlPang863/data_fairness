import json
import numpy as np
import os
import shutil
import tensorflow as tf
import random
import torch
from types import SimpleNamespace
import pandas as pd
from datetime import datetime
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
  random.seed(seed)
  torch.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
  if torch.cuda.is_available():
      torch.cuda.manual_seed_all(seed)

########################################################################################################################
#  print
########################################################################################################################

def print_stats(t, T, t_incr, t_tot, train_metric, test_metric, init=False, is_val = False, metric = 'dp'):
  prog = t / T * 100
  if is_val:
    val = 'val'
  else:
    val = 'test'
  # print(f'{prog:6.2f}% | time: {t_incr:5.1f}s ({t_tot/60:5.1f}m) | step: {t:6d} |',
  #         f"train acc: {train_metric['accuracy']:.3f}({train_metric['acc'][0]:.3f}, {train_metric['acc'][1]:.3f}) | train ar gap: {abs(train_metric['ar'][0]-train_metric['ar'][1]):.3f} | train loss: {train_metric['loss']:.3f} |{val} acc: {test_metric['accuracy']:.3f} ({test_metric['acc'][0]:.3f}, {test_metric['acc'][1]:.3f}) | {val} ar gap: {abs(test_metric['ar'][0] - test_metric['ar'][1]):.4f}")
  if metric == 'dp':
    print(f'{prog:6.2f}% | time: {t_incr:5.1f}s ({t_tot/60:5.1f}m) | step: {t:6d} |',
          f"train acc: {train_metric['accuracy']:.3f} | train loss: {train_metric['loss']:.3f} |{val} acc: {test_metric['accuracy']:.3f} ({test_metric['acc'][0]:.3f}, {test_metric['acc'][1]:.3f}) | {val} dp gap: {abs(test_metric['ar'][0] - test_metric['ar'][1]):.4f}")
  elif metric == 'eop':
    print(f'{prog:6.2f}% | time: {t_incr:5.1f}s ({t_tot/60:5.1f}m) | step: {t:6d} |',
          f"train acc: {train_metric['accuracy']:.3f} | train loss: {train_metric['loss']:.3f} |{val} acc: {test_metric['accuracy']:.3f} ({test_metric['acc'][0]:.3f}, {test_metric['acc'][1]:.3f}) | {val} eop gap: {abs(test_metric['tpr'][0] - test_metric['tpr'][1]):.4f}")
  elif metric == 'eod':
    print(f'{prog:6.2f}% | time: {t_incr:5.1f}s ({t_tot/60:5.1f}m) | step: {t:6d} |',
          f"train acc: {train_metric['accuracy']:.3f} | train loss: {train_metric['loss']:.3f} |{val} acc: {test_metric['accuracy']:.3f} ({test_metric['acc'][0]:.3f}, {test_metric['acc'][1]:.3f}) | {val} eod gap: {(abs(test_metric['tpr'][0] - test_metric['tpr'][1]) + abs(test_metric['fpr'][0] - test_metric['fpr'][1]))/2.0:.4f}")

def date_from_str(s):
    return datetime.strptime(s, '%Y-%m-%d %H:%M:%S')


def preprocess_compas(filename = "compas-scores-two-years.csv"):
# Pre-process the compas data
# Input: Raw COMPAS data (filename)
# Output: N*d pd.DataFrame 
#         N: number of instances 
#         d: feature dimension
# sensitive attribute: race, sex
# model prediction: decile_score
# ground-truth label: is_recid or two_year_recid

    raw_data = pd.read_csv(f'./data/COMPAS/{filename}')
    # print(f'Num rows (total): {len(raw_data)}')

    # remove missing data
    df = raw_data[((raw_data['days_b_screening_arrest'] <=30) & 
        (raw_data['days_b_screening_arrest'] >= -30) &
        (raw_data['is_recid'] != -1) &
        (raw_data['c_charge_degree'] != 'O') & 
        (raw_data['score_text'] != 'N/A')
        )]
    # length of staying in jail
    df['length_of_stay'] = (df['c_jail_out'].apply(date_from_str) - df['c_jail_in'].apply(date_from_str)).dt.total_seconds()


    sel_columns = [ 'first', 'last',  'name',
                    'age',  'age_cat', 'race',  'sex',  
                    'decile_score', 'score_text', 'v_decile_score',  
                    'is_recid', 'two_year_recid', 
                    'priors_count', 'days_b_screening_arrest','length_of_stay', 'c_charge_degree']

    df = df[sel_columns]
    return df


def race_encode(df):

    func_encode = race_encode_compas
    # print('Before encoding:')
    # print(df.race.value_counts())
    df.race = df.race.apply(func_encode)
    # print('After encoding:')
    # print(df.race.value_counts())
    # print('\n')

    return df


def race_encode_compas(s):
# COMPAS mapping: (combine 3 and 4 due to the sample size)
# Race                  #
# African-American    3175
# Caucasian           2103
# Hispanic             509
# Other                343
# Asian                 31
# Native American       11
# African-American          --> Black       --> 0
# Caucasian                 --> White       --> 1
# Hispanic                  --> Hispanic    --> 2
# Asian                     --> Asian       --> 3
# Other, Native American    --> Other       --> 4
    race_dict = {'African-American':0,'Caucasian':1, 'Hispanic':2}
    return race_dict.get(s, 3)
