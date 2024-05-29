
import jax
# import tensorflow as tf
from jax import jacrev, numpy as jnp
import numpy as np
import time


from .data import  load_celeba_dataset_torch, preprocess_func_celeba_torch, load_data, gen_preprocess_func_torch2jax
from .models import get_model
from .recorder import init_recorder, record_train_stats, save_recorder, record_test, save_checkpoint,load_checkpoint
import pdb

from .hoc_fairlearn import *
from .train_state import test_step, get_train_step, create_train_state, infl_step, infl_step_fair, infl_step_per_sample, train_plain
from .metrics import compute_metrics, compute_metrics_fair
from .utils import set_global_seed, make_dirs, log_and_save_args
from . import global_var
import collections
import os

import logging
from .loss_func import *

from typing import Any,Callable
from scipy.special import xlogy

from jax.flatten_util import ravel_pytree
from jax import jacfwd, jacrev, jit, vmap
from jax.tree_util import tree_flatten, tree_map


os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"   # This disables the preallocation behavior. JAX will instead allocate GPU memory as needed, potentially decreasing the overall memory usage.


def test(args, state, data):
  """
  Test
  """
  logits, labels, groups = [], [], []
  preprocess_func_torch2jax = gen_preprocess_func_torch2jax(args.dataset)
  for example in data:
    # batch = preprocess_func_celeba(example, args)
    batch = preprocess_func_torch2jax(example, args, noisy_attribute = None)
    # batch = example
    logit= test_step(state, batch)
    logits.append(logit)
    labels.append(batch['label'])
    groups.append(batch['group'])

  return compute_metrics_fair(
    logits=jnp.concatenate(logits),
    labels=jnp.concatenate(labels),
    groups=jnp.concatenate(groups),
  )
  # return None

def fair_train_validation(args):
    ########################################################################################################
  # setup
  set_global_seed(args.train_seed)
#   make_dirs(args)
  preprocess_func_torch2jax = gen_preprocess_func_torch2jax(args.dataset)


    #   if args.strategy == 1:
    #     [_, _], part1, part2 = load_celeba_dataset_torch(args, shuffle_files=True, split='train', batch_size=args.train_batch_size, ratio = args.label_ratio, sampled_idx=[], return_part2=True)
    #     load_name = f'./results/s2_{args.metric}_{args.label_ratio}_new{args.new_data_each_round}_100round_case1_remove_unfair_trainConf{args.train_conf}_posloss{args.remove_pos}_poslossOrg{args.remove_posOrg}.npy'
    #     indices = np.load(load_name, allow_pickle=True)
    #     sel_idx = list(indices[args.sel_round][2])
    #     num_sample_to_add = len(sel_idx) - len(part1)
    #     random.Random(args.train_seed).shuffle(part2)
    #     sel_idx = part1 + part2[:num_sample_to_add]
    #     print(f'randomly select {len(part1)} + {num_sample_to_add} = {len(sel_idx)} samples')
    #   elif args.strategy in [2,3,4,5]:
    #     load_name = f'./results/s{args.strategy}_{args.metric}_{args.label_ratio}_new{args.new_data_each_round}_100round_case1_remove_unfair_trainConf{args.train_conf}_posloss{args.remove_pos}_poslossOrg{args.remove_posOrg}.npy'
    #     indices = np.load(load_name, allow_pickle=True)
    #     sel_idx = list(indices[args.sel_round][2])
    #   elif args.strategy == 6:
    #     [_, _], sel_idx = load_celeba_dataset_torch(args, shuffle_files=True, split='train', batch_size=args.train_batch_size, ratio = args.label_ratio, sampled_idx=None, fair_train=True) # use all data for training
    #   else:
    #     raise NameError('We only have strategies from 1 to 6')

    #   [train_loader_labeled, _], _ = load_celeba_dataset_torch(args, shuffle_files=True, split='train', batch_size=args.train_batch_size, ratio = 0.0, sampled_idx=sel_idx, fair_train=True)

    
    #   [val_loader, test_loader], _ = load_celeba_dataset_torch(args, shuffle_files=True, split='test', batch_size=args.test_batch_size, ratio = args.val_ratio, fair_train=True)

    #   args.image_shape = args.img_size
    
  new_labels = {}
  train_loader_labeled, train_loader_unlabeled, idx_with_labels = load_data(args, args.dataset, mode = 'train', aux_dataset=args.aux_data)
  _, train_loader_unlabeled_org, idx_with_labels_org = load_data(args, args.dataset, mode = 'train', aux_dataset=None)
  args.train_with_org = True


  val_loader, test_loader = load_data(args, args.dataset, mode = 'val')


  preprocess_func_torch2jax = gen_preprocess_func_torch2jax(args.dataset)
  
########################################################################################################
  # setup
  load_checkpoint_model = True
  if load_checkpoint_model:
    # dict_keys(['step', 'params', 'opt_state', 'batch_stats'])
    checkpoint = load_checkpoint(args.save_dir + '/ckpts')
    # pdb.set_trace()
    model = get_model(args)
    # args.hidden_size = model_linear.hidden_size
    # 获取模型状态和其他信息
    state = create_train_state(model, args, checkpoint['params'],return_opt=False)
    
  else:
  
    model = get_model(args)
    # args.hidden_size = model_linear.hidden_size
    state = create_train_state(model, args, return_opt=False)
  ########################################################################################################

  # get model size
  flat_tree = jax.tree_util.tree_leaves(state.params)
  num_layers = len(flat_tree)
  print(f'Numer of layers {num_layers}')

  rec = init_recorder()

  

  # info
  log_and_save_args(args)
  time_start = time.time()
  time_now = time_start
  print('train net...')

  # begin training
  lmd = args.lmd


  conf = 'no_conf' # warm up
  global_var.set_value('args', args)


  train_step, train_step_warm = get_train_step(args.method)
  worst_group_id = 0
  for epoch_i in range(args.num_epochs):

    args.curr_epoch = epoch_i
    t = 0
    num_sample_cur = 0
    print(f'Epoch {epoch_i}')
    val_iter = iter(val_loader)
     
    #set to the validation set size
    # temp = args.datasize
    # # args.datasize = int(args.datasize * 4 * args.val_ratio * args.label_ratio)
    args.datasize = len(val_loader.dataset)
    # pdb.set_trace()  
    while t * args.train_batch_size < args.datasize:
    # for example in train_loader_labeled:
        for example in val_loader:    #### use validation data to train models #####

            try:
                # Samples the batch
                example_fair = next(val_iter)
            except StopIteration:
                # restart the generator if the previous generator is exhausted.
                val_iter = iter(val_loader)
                example_fair = next(val_iter)

            bsz = example[0].shape[0]

            num_sample_cur += bsz
            batch = preprocess_func_torch2jax(example, args, noisy_attribute = None)
            batch_fair = preprocess_func_torch2jax(example_fair, args, noisy_attribute = None)
            t += 1
            if t * args.train_batch_size > args.datasize:
                break

            # train
            if args.method == 'plain':
                state, train_metric = train_step(state, batch)
            elif args.method in ['dynamic_lmd']:
                if state.step >= args.warm_step:
                    state, train_metric, train_metric_fair, lmd = train_step(state, batch, batch_fair, lmd = lmd, T=None, worst_group_id = worst_group_id)
                else:
                    state, train_metric, _, lmd = train_step_warm(state, batch, batch_fair, lmd = lmd, T=None, worst_group_id = worst_group_id)
                # print(f'warm up step {state.step}/{args.warm_step}')
                
            else:
                raise NameError('Undefined optimization mechanism')

            rec = record_train_stats(rec, t-1, train_metric, 0)

            if t % args.log_steps == 0:
                # test
                # epoch_pre = epoch_i
            #   pdb.set_trace()
            #   print(f'[Step {state.step}] Conf: {args.conf} Current lr is {np.round(lr_scheduler(state.step), 5)}')
                print(f'[Step {state.step}] Conf: {args.conf}')

                test_metric = test(args, state, test_loader)
                val_metric = test(args, state, val_loader)
                worst_group_id = np.argmin(val_metric['acc'])
                _, time_now = record_test(rec, t+args.datasize*epoch_i//args.train_batch_size, args.datasize*args.num_epochs//args.train_batch_size, time_now, time_start, train_metric, test_metric, val_metric=val_metric)
                rec, time_now = record_test(rec, t+args.datasize*epoch_i//args.train_batch_size, args.datasize*args.num_epochs//args.train_batch_size, time_now, time_start, train_metric, test_metric)

                print(f'lmd is {lmd}')

    rec = save_checkpoint(args.save_dir, t+args.datasize*epoch_i//args.train_batch_size, state, rec, save=False) #在 save_checkpoint 函数中设置 save=True 参数，以确保实际保存模型的文件

  # wrap it up
#   file_name = f'/s{args.strategy}_{args.metric}_{args.label_ratio}_new{args.new_data_each_round}_100round_case1_remove_unfair_trainConf{args.train_conf}_posloss{args.remove_pos}_poslossOrg{args.remove_posOrg}_{args.sel_round}.pkl'
#   save_recorder(args.save_dir, rec, file_name=file_name)
