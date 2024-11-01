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
import warnings
from sklearn.preprocessing import LabelEncoder
import re

########################################################################################################################
#  Argument Management
########################################################################################################################

def log_and_save_args(args):
    """Logs and saves arguments to JSON."""
    print('train args:')
    print_args(args)
    save_args(args, args.save_dir, verbose=True)

def save_args(args, save_dir, verbose=True):
    save_path = os.path.join(save_dir, 'args.json')
    with open(save_path, 'w') as f:
        json.dump(vars(args), f, indent=4)
    if verbose:
        print(f'Saved args to {save_path}')
    return save_path

def load_args(load_dir, verbose=True):
    load_path = os.path.join(load_dir, 'args.json')
    with open(load_path, 'r') as f:
        args = SimpleNamespace(**json.load(f))
    if verbose:
        print(f'Loaded args from {load_path}')
    return args

def print_args(args):
    print(json.dumps(vars(args), indent=4))

########################################################################################################################
#  File Management
########################################################################################################################

def make_dir(path):
    """Creates or overwrites a directory."""
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def make_dirs(args):
    make_dir(args.save_dir)
    make_dir(os.path.join(args.save_dir, 'ckpts'))

########################################################################################################################
#  Seed Setting
########################################################################################################################

def set_global_seed(seed=0):
    """Sets the global seed across libraries for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

########################################################################################################################
#  Logging
########################################################################################################################

def print_stats(t, T, t_incr, t_tot, train_metric, test_metric, init=False, is_val=False, metric='dp', warm=False):
    """Prints progress and metric stats."""
    prog = t / T * 100
    val = 'warm_' + ('val' if is_val else 'test') if warm else ('val' if is_val else 'test')
    if metric == 'dp':
        print(f'{prog:6.2f}% | time: {t_incr:5.1f}s ({t_tot/60:5.1f}m) | step: {t:6d} |',
              f"train acc: {train_metric['accuracy']:.3f} | train loss: {train_metric['loss']:.3f} | {val} f1_score: {test_metric['f1_score']:.3f} | {val} acc: {test_metric['accuracy']:.3f} ({test_metric['acc'][0]:.3f}, {test_metric['acc'][1]:.3f}) | {val} dp gap: {abs(test_metric['ar'][0] - test_metric['ar'][1]):.4f}")
    elif metric == 'eop':
        print(f'{prog:6.2f}% | time: {t_incr:5.1f}s ({t_tot/60:5.1f}m) | step: {t:6d} |',
              f"train acc: {train_metric['accuracy']:.3f} | train loss: {train_metric['loss']:.3f} | {val} f1_score: {test_metric['f1_score']:.3f} | {val} acc: {test_metric['accuracy']:.3f} | {val} eop gap: {abs(test_metric['tpr'][0] - test_metric['tpr'][1]):.4f}")
    elif metric == 'eod':
        print(f'{prog:6.2f}% | time: {t_incr:5.1f}s ({t_tot/60:5.1f}m) | step: {t:6d} |',
              f"train acc: {train_metric['accuracy']:.3f} | train loss: {train_metric['loss']:.3f} | {val} f1_score: {test_metric['f1_score']:.3f} | {val} acc: {test_metric['accuracy']:.3f} | {val} eod gap: {(abs(test_metric['tpr'][0] - test_metric['tpr'][1]) + abs(test_metric['fpr'][0] - test_metric['fpr'][1]))/2.0:.4f}")

def date_from_str(s):
    return datetime.strptime(s, '%Y-%m-%d %H:%M:%S')

########################################################################################################################
#  Data Preprocessing
########################################################################################################################

def preprocess_compas(filename="compas-scores-two-years.csv"):
    """Preprocess COMPAS dataset: filters data and computes additional features."""
    raw_data = pd.read_csv(f'./data/COMPAS/{filename}')
    df = raw_data[(raw_data['days_b_screening_arrest'].between(-30, 30)) &
                  (raw_data['is_recid'] != -1) &
                  (raw_data['c_charge_degree'] != 'O') &
                  (raw_data['score_text'] != 'N/A')]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df['length_of_stay'] = (df['c_jail_out'].apply(date_from_str) - df['c_jail_in'].apply(date_from_str)).dt.total_seconds()

    selected_columns = [
        'age', 'age_cat', 'race', 'sex', 'decile_score', 'is_recid', 'c_charge_degree',
        'two_year_recid', 'priors_count', 'days_b_screening_arrest', 'length_of_stay'
    ]
    return df[selected_columns]





def race_encode(df):
    """Encodes race using COMPAS-specific mapping."""
    race_dict = {'African-American': 0, 'Caucasian': 1, 'Hispanic': 2}
    df.race = df.race.apply(lambda x: race_dict.get(x, 3))
    return df

def preprocess_adult():
    """Preprocesses the UCI Adult dataset by loading, cleaning, and encoding categorical features."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    column_names = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
        'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
        'hours-per-week', 'native-country', 'income'
    ]
    df = pd.read_csv(url, names=column_names, sep=r'\s*,\s*', engine='python').dropna()
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    return df

def preprocess_jigsaw(df):
    """Preprocesses the Jigsaw dataset by cleaning text data."""
    df['comment_text'] = df['comment_text'].str.lower() \
                            .apply(lambda x: re.sub(r'http\S+', '', x)) \
                            .apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x))
    return df
