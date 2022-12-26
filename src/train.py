import jax
# import tensorflow as tf
from jax import numpy as jnp
import numpy as np
import time
from .data import  load_celeba_dataset_torch, preprocess_func_celeba_torch
from .models import get_model
from .recorder import init_recorder, record_train_stats, save_recorder, record_test, save_checkpoint
import pdb
from .hoc_fairlearn import *
from .train_state import test_step, get_train_step, create_train_state, infl_step
from .metrics import compute_metrics
from .utils import set_global_seed, make_dirs, log_and_save_args
from . import global_var
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"   # This disables the preallocation behavior. JAX will instead allocate GPU memory as needed, potentially decreasing the overall memory usage.
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".20" # If preallocation is enabled, this makes JAX preallocate XX% of currently-available GPU memory, instead of the default 90%.

# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform" # minimal GPU memory, very slow
########################################################################################################################


     
# def get_T_p(config, noisy_attribute, lr = 0.1, true_attribute = None):

#   # Estimate T and P with HOC
#   T_est, p_est = get_T_global_min(config, noisy_attribute, lr = lr)
#   print(f'\n\n-----------------------------------------')
#   print(f'Estimation finished!')
#   # np.set_printoptions(precision=1)
#   print(f'The estimated T (*100) is \n{np.round(T_est*100,1)}')
#   print(f'The estimated p (*100) is \n{np.round(p_est*100,1)}')
#   if true_attribute is not None:
#       T_true, p_true = check_T(KINDS=config.num_classes, clean_label=true_attribute, noisy_label=noisy_attribute)
#       # print(f'T_inv: \nest: \n{np.linalg.inv(T_est)}\ntrue:\n{np.linalg.inv(T_true)}')
#       print(f'T_true: {T_true},\n T_est: {T_est}')
#       print(f'p_true: {p_true},\n p_est: {p_est}')
#   return T_est, p_est, T_true, p_true.reshape(-1,1)

def sample_by_infl(args, state, val_data, unlabeled_data, num):
  """
  Get influence score of each unlabeled_data on val_data, then sample according to scores
  """
  print('begin calculating influence')
  num_samples = 0.0
  grad_sum = 0.0
  for example in val_data: # Need to run on the validation dataset to avoid the negative effect of distribution shift, e.g., DP is not robust to distribution shift. For fairness, val data may be iid as test data 
    batch = preprocess_func_celeba_torch(example, args)
    grads_each_sample = infl_step(state, batch)

    # moving average
    grad_sum += np.sum(grads_each_sample, axis=0)
    num_samples += grads_each_sample.shape[0]
    # grad_sum /= num_samples
  grad_avg = (grad_sum/num_samples).reshape(-1,1)

  # check unlabeled data
  score = []
  for example in unlabeled_data:
    batch = preprocess_func_celeba_torch(example, args)
    grads_each_sample = infl_step(state, batch)
    score = np.matmul(grads_each_sample, grad_avg) # bsz * 1
    pdb.set_trace()
    # TODO

  print('calculating influence -- done')






def test(args, state, data):
  """
  Test
  """
  logits, labels, groups = [], [], []
  for example in data:
    # batch = preprocess_func_celeba(example, args)
    batch = preprocess_func_celeba_torch(example, args, noisy_attribute = None)
    # batch = example
    logit= test_step(state, batch)
    logits.append(logit)
    labels.append(batch['label'])
    groups.append(batch['group'])

  return compute_metrics(
    logits=jnp.concatenate(logits),
    labels=jnp.concatenate(labels),
    groups=jnp.concatenate(groups),
  )
  # return None

def train(args):
  # setup
  set_global_seed(args.train_seed)
  make_dirs(args)

  train_loader_labeled, train_loader_unlabeled = load_celeba_dataset_torch(args, shuffle_files=True, split='train', batch_size=args.train_batch_size, ratio = args.label_ratio)
  val_loader, test_loader = load_celeba_dataset_torch(args, shuffle_files=False, split='test', batch_size=args.test_batch_size, ratio = args.val_ratio)

  args.image_shape = args.img_size
  # setup
  model, model_linear = get_model(args)
  args.hidden_size = model_linear.hidden_size
  state = create_train_state(model, args)


  rec = init_recorder()

  

  # info
  log_and_save_args(args)
  time_start = time.time()
  time_now = time_start
  print('train net...')

  # begin training
  lmd = args.lmd
  loss = 0.0
  loss_rec = [loss]
  train_step = get_train_step(args.method)
  # epoch_pre = 0
  for epoch_i in range(args.num_epochs):


    t = 0
    num_sample_cur = 0
    print(f'Epoch {epoch_i}')
    # if epoch_i < args.warm_epoch: 
    #   print(f'warmup epoch = {epoch_i+1}/{args.warm_epoch}')
    # if epoch_i == args.warm_epoch:
    #   state_reg = create_train_state(model, args, params=state.params) # use the full model
    while t * args.train_batch_size < args.datasize:
      for example in train_loader_labeled:
        # pdb.set_trace()
        bsz = example[0].shape[0]

        num_sample_cur += bsz
        example = preprocess_func_celeba_torch(example, args, noisy_attribute = None)
        t += 1
        if t * args.train_batch_size > args.datasize:
          break
        # load data
        if args.balance_batch:
          image, group, label = example['feature'], example['group'], example['label']
          num_a, num_b = jnp.sum((group == 0) * 1.0), jnp.sum((group == 1) * 1.0)
          min_num = min(num_a, num_b).astype(int)
          total_idx = jnp.arange(len(group))
          if min_num > 0:
            group_a = total_idx[group == 0]
            group_b = total_idx[group == 1]
            group_a = group_a.repeat(args.train_batch_size//2//len(group_a)+1)[:args.train_batch_size//2]
            group_b = group_b.repeat(args.train_batch_size//2//len(group_b)+1)[:args.train_batch_size//2]

            sel_idx = jnp.concatenate((group_a,group_b))
            batch = {'feature': jnp.array(image[sel_idx]), 'label': jnp.array(label[sel_idx]), 'group': jnp.array(group[sel_idx])}
          else:
            print(f'current batch only contains one group')
            continue

        else:
          batch = example

        # train
        if args.method == 'plain':
          state, train_metric = train_step(state, batch)
        elif args.method in ['fix_lmd','dynamic_lmd']:
          state, train_metric, lmd = train_step(state, batch, lmd = lmd, T=None)
        else:
          raise NameError('Undefined optimization mechanism')

        rec = record_train_stats(rec, t-1, train_metric, 0)
      
        if t % args.log_steps == 0:
          # test
          # epoch_pre = epoch_i
          test_metric = test(args, state, test_loader)
          rec, time_now = record_test(rec, t+args.datasize*epoch_i//args.train_batch_size, args.datasize*args.num_epochs//args.train_batch_size, time_now, time_start, train_metric, test_metric)
          if epoch_i > args.warm_epoch:
            # infl 
            sampled_idx = sample_by_infl(args, state, val_loader, train_loader_unlabeled, num = args.new_data_each_round)

            train_loader_labeled, train_loader_unlabeled = load_celeba_dataset_torch(args, shuffle_files=True, split='train', batch_size=args.train_batch_size, ratio = args.label_ratio, sampled_idx=sampled_idx)

          
          print(f'lmd is {lmd}')



    rec = save_checkpoint(args.save_dir, t+args.datasize*epoch_i//args.train_batch_size, state, rec, save=False)

  # wrap it up
  save_recorder(args.save_dir, rec)
  # return test_metric
