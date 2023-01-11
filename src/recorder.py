import pickle
from types import SimpleNamespace
import time
from .utils import print_stats

def init_recorder():
  rec = SimpleNamespace()
  rec.train_step = []
  rec.train_loss = []
  rec.train_acc = []
  rec.train_acc_pair = []
  rec.train_ar = []
  rec.train_tpr = []
  rec.train_fpr = []
  rec.lr = []
  rec.test_step = []
  rec.test_loss = []
  rec.test_acc = []
  rec.test_acc_pair = []
  rec.test_ar = []
  rec.test_tpr = []
  rec.test_fpr = []
  rec.ckpts = []
  rec.T = []
  rec.label_pred = []
  rec.label_noisy = []
  rec.label_clean = []
  rec.label_org = []
  rec.label_org_pred = []
  return rec


def record_train_stats(rec, step, metric, lr):
  rec.train_step.append(step)
  rec.train_loss.append(metric['loss'])
  rec.train_acc.append(metric['accuracy'])
  rec.train_acc_pair.append(metric['acc'])
  rec.train_ar.append(metric['ar'])
  rec.train_tpr.append(metric['tpr'])
  rec.train_fpr.append(metric['fpr'])
  rec.lr.append(lr)
  return rec


def record_test_stats(rec, step, metric):
  rec.test_step.append(step)
  rec.test_loss.append(metric['loss'])
  rec.test_acc.append(metric['accuracy'])
  rec.test_acc_pair.append(metric['acc'])
  rec.test_ar.append(metric['ar'])
  rec.test_tpr.append(metric['tpr'])
  rec.test_fpr.append(metric['fpr'])
  return rec


def record_ckpt(rec, step):
  rec.ckpts.append(step)
  return rec


def save_recorder(save_dir, rec, verbose=True, file_name = None):
  if file_name:
    save_path = save_dir + file_name
  else:
    save_path = save_dir + '/recorder.pkl'

  with open(save_path, 'wb') as f: pickle.dump(vars(rec), f)
  if verbose: print(f'Save record to {save_path}')
  return save_path


def load_recorder(load_dir, verbose=True):
  load_path = load_dir + '/recorder.pkl'
  with open(load_path, 'rb') as f: rec = SimpleNamespace(**pickle.load(f))
  if verbose: print(f'Load record from {load_path}')
  return rec


def record_test(rec, t, T, t_prev, t_start, train_metric, test_metric, init=False, val_metric = None):
  rec = record_test_stats(rec, t, test_metric)
  t_now = time.time()
  t_incr, t_tot = t_now - t_prev, t_now - t_start
  
  if val_metric:
    print_stats(t, T, t_incr, t_tot, train_metric, val_metric, init, is_val=True)
  else:
    print_stats(t, T, t_incr, t_tot, train_metric, test_metric, init, is_val=False)



  return rec, t_now


# checkpoint
from flax.training import checkpoints
def save_checkpoint(save_dir, step, state, rec, save=True):
  if save:
    checkpoints.save_checkpoint(save_dir + '/ckpts', target = state, step = step, keep=1)
  rec = record_ckpt(rec, step)
  return rec