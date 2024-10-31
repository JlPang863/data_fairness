
import jax
# import tensorflow as tf
from jax import jacrev, numpy as jnp
import numpy as np
import time
import random

from .data import  load_celeba_dataset_torch, preprocess_func_celeba_torch, load_data, gen_preprocess_func_torch2jax
from .models import get_model
from .recorder import init_recorder, record_train_stats, save_recorder, record_test, save_checkpoint,load_checkpoint

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

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


def compute_hessian(state, train_loader_labeled, val_data):

  args = global_var.get_value('args')
  preprocess_func_torch2jax = gen_preprocess_func_torch2jax(args.dataset)
  num_samples = 0.0
  grad_sum = 0.0
  grad_org_sum = 0.0
  for example in val_data: # Need to run on the validation dataset to avoid the negative effect of distribution shift, e.g., DP is not robust to distribution shift. For fairness, val data may be iid as test data 
    batch = preprocess_func_torch2jax(example, args)
    bsz = batch['feature'].shape[0]
    grads, grads_org = np.asarray(infl_step(state, batch))

    # sum & average
    grad_sum += grads * bsz
    grad_org_sum += grads_org * bsz
    num_samples += bsz

  grad_avg = (grad_sum/num_samples).reshape(-1,1)
  grad_org_avg = (grad_org_sum/num_samples).reshape(-1,1)

  example_size = 10 if args.strategy !=8 else 1

  H_v_tmp = grad_avg
  H_v_org_tmp = grad_org_avg

  for example in train_loader_labeled:
    if example_size == 0:
      break
    
    batch_labeled = preprocess_func_torch2jax(example, args)

    grads_each_sample, logits = infl_step_per_sample(state, batch_labeled)
    #compute the gradient of labeled dataset
    for idx in range(grads_each_sample.shape[0]):
      grad_sample = grads_each_sample[idx].reshape(-1,1)
      H_v_tmp = grad_avg + (H_v_tmp - (np.dot(grad_sample.T, H_v_tmp) * grad_sample))
      H_v_org_tmp = grad_org_avg + (H_v_org_tmp - np.dot(grad_sample.T,H_v_org_tmp) * grad_sample)

    example_size -= 1

  return H_v_tmp, H_v_org_tmp

def compute_bald_score(args, predictions):
    """
    Compute the score according to the heuristic.

    Args:
        predictions (ndarray): Array of predictions

    Returns:
        Array of scores.
    """
    assert predictions.ndim == 2
    entropy = -np.sum(xlogy(predictions, predictions), axis=1)
    return entropy
    

def jtt_misclassified_examples(args, model, state, train_loader, num):
    """
    Identifies misclassified examples in the training set.

    Args:
        model (Model): The trained model.
        state (TrainState): Current training state.
        train_loader (DataLoader): DataLoader for the training set.
        num_samples (int): Number of samples to identify.

    Returns:
        list: Indices of misclassified examples.
        dict: Empty dictionary (as placeholders for new labels).
    """
    misclassified_indices = []
    misclassified_indices_0 = []  # misclassified example's label 0
    misclassified_indices_1 = []  # misclassified example's label 1
    for example in train_loader:
      # Get predictions

      preprocess_func_torch2jax = gen_preprocess_func_torch2jax(args.dataset)
      example = preprocess_func_torch2jax(example, args)

      features = example['feature']
      labels = example['label']
      features = jnp.array(features)
      labels = jnp.array(labels)

      #prediction
      output = state.apply_fn({'params': state.params}, features)
      logits = output if not isinstance(output, tuple) else output[0]
      y_pred = logits.argmax(axis=1)
      
    #  Identify misclassified samples
      misclassified = np.where(y_pred != labels)[0]

      for idx in misclassified:
        if labels[idx] == 0 and len(misclassified_indices_0) < num//2:
            misclassified_indices_0.append(idx)
        elif labels[idx] == 1 and len(misclassified_indices_1) < num//2:
            misclassified_indices_1.append(idx)

      # check the number of misclassified examples
      if len(misclassified_indices_0) >= num//2 and len(misclassified_indices_1) >= num//2:
          break

      balanced_misclassified_indices = misclassified_indices_0 + misclassified_indices_1

      random.shuffle(balanced_misclassified_indices)

      return balanced_misclassified_indices, {}

def sample_by_infl(args, state, val_data, unlabeled_data, num, force_org = False):
  """
  Get influence score of each unlabeled_data on val_data, then sample according to scores
  For fairness, the sign is very important
  """
  preprocess_func_torch2jax = gen_preprocess_func_torch2jax(args.dataset)
  if args.aux_data is None or force_org:
    preprocess_func_torch2jax_aux = gen_preprocess_func_torch2jax(args.dataset)
  else:
    preprocess_func_torch2jax_aux = gen_preprocess_func_torch2jax(args.aux_data)
  print('begin calculating influence')
  num_samples = 0.0
  grad_sum = 0.0
  grad_org_sum = 0.0
  grad_fair_sum = 0.0
  for example in val_data: # Need to run on the validation dataset to avoid the negative effect of distribution shift, e.g., DP is not robust to distribution shift. For fairness, val data may be iid as test data 
    batch = preprocess_func_torch2jax(example, args)
    bsz = batch['feature'].shape[0]
    grads, grads_org = np.asarray(infl_step(state, batch))

    grads_fair_batch = np.asarray(infl_step_fair(state, batch))

    # sum & average
    grad_sum += grads * bsz
    grad_org_sum += grads_org * bsz
    grad_fair_sum += grads_fair_batch * bsz
    num_samples += bsz

  grad_avg = (grad_sum/num_samples).reshape(-1,1)
  grad_org_avg = (grad_org_sum/num_samples).reshape(-1,1)
  grad_fair = (grad_fair_sum/num_samples).reshape(-1,1)

  # check unlabeled data
  score = []
  score_before_check = []
  score_org = []
  idx = []
  expected_label = []
  true_label = []
  for example in unlabeled_data:
    batch = preprocess_func_torch2jax_aux(example, args)
    batch_unlabeled = batch.copy()
    batch_unlabeled['label'] = None # get grad for each label. We do not know labels of samples in unlabeled data
    grads_each_sample, logits = infl_step_per_sample(state, batch_unlabeled)
    grads_each_sample = np.asarray(grads_each_sample)
    logits = np.asarray(logits)

    '''
    calculate the influence of prediction/fairness component
    '''
    infl = - np.matmul(grads_each_sample, grad_avg) # new_loss - cur_los  # 
    infl_org = - np.matmul(grads_each_sample, grad_org_avg) # new_loss - cur_los  # 
    infl_fair = - np.matmul(grads_each_sample, grad_fair)

    # ---------- Sampling strategy -------------------
    # Strategy 1 (baseline): random
    if args.strategy == 1:
      score += [1] * batch['label'].shape[0]
    # Strategy 2 (idea 1): find the label with least absolute influence, then find the sample with most negative fairness infl
    elif args.strategy == 2:
      label_expected = np.argmin(abs(infl), 1).reshape(-1)

    # Strategy 3 (idea 2): find the label with minimal influence values (most negative), then find the sample with most negative infl 
    elif args.strategy == 3:
      label_expected = np.argmin(infl, 1).reshape(-1)

    # Strategy 4: use true label
    elif args.strategy == 4:
      label_expected = batch['label'].reshape(-1)

    # Strategy 4: use model predicted label
    elif args.strategy == 5:
      label_expected = np.argmax(logits, 1).reshape(-1)

    # Strategy 6: baseline 1 - BALD
    elif args.strategy == 6:
      label_expected = np.argmax(logits, 1).reshape(-1)
      
    ##skip strategy 7 JTT baseline here since JTT does not depends on influence scores

    # Strategy 8: Influence selection for active learning (ISAL), select unlabeled samples according to influence scores
    elif args.strategy == 8:
      # use model predicted label as pseuod-labels
      label_expected = np.argmax(logits, 1).reshape(-1)

    ########################################################
    ################# sample selection #####################
    ########################################################
    if args.strategy > 1 and args.strategy < 6:
      score_tmp = (infl_fair[range(infl_fair.shape[0]), label_expected]).reshape(-1)
      score_org += score_tmp.tolist()
      infl_fair_true = infl_fair[range(infl_fair.shape[0]), batch['label'].reshape(-1)].reshape(-1)
      if args.remove_pos:
        infl_true = infl[range(infl.shape[0]), batch['label'].reshape(-1)].reshape(-1) # # case1_remove_posloss
        infl_expected = infl[range(infl.shape[0]), label_expected].reshape(-1)
      if args.remove_posOrg:
        infl_true = infl_org[range(infl_org.shape[0]), batch['label'].reshape(-1)].reshape(-1) # case1_remove_poslossOrg
        infl_expected = infl_org[range(infl_org.shape[0]), label_expected].reshape(-1)
    
      if args.remove_pos or args.remove_posOrg:
        score_tmp[infl_expected > 0] = 0.0
        score_before_check += score_tmp.tolist()

        score_tmp[infl_true > 0] = 0.0 # remove_posloss or remove_poslossOrg

      score_tmp[infl_fair_true > 0] = 0.0 # remove_unfair, use true label

      score += score_tmp.tolist()

      expected_label += label_expected.tolist()
      true_label += batch['label'].tolist()

    #baseline: BALD
    elif args.strategy == 6:

      score_tmp = -1 * compute_bald_score(args, softmax(logits))
      score += score_tmp.tolist()
      expected_label += label_expected.tolist()
      true_label += batch['label'].tolist()

    elif args.strategy == 8:

      infl = - np.matmul(grads_each_sample, args.H_v) # new_loss - cur_los  # 
      infl_org = - np.matmul(grads_each_sample, args.H_v_org) # new_loss - cur_los  # 



      score_tmp = (infl[range(infl.shape[0]), label_expected].reshape(-1))
      score += score_tmp.tolist()
      expected_label += label_expected.tolist()
      true_label += batch['label'].tolist()


    ########################################################
    ################# sample selection #####################
    ########################################################
      

    idx += batch['index'].tolist()
    if len(score) >= num * 100: # 100
      break

  if args.strategy == 1:
    sel_idx = list(range(len(score)))
    random.Random(args.infl_random_seed).shuffle(sel_idx)
    sel_idx = sel_idx[:num]
    sel_true_false_with_labels = sel_idx

  # Strategy 2--8
  else:
    sel_idx = np.argsort(score)[:num]
    max_score = score[sel_idx[-1]]
    if max_score >= 0.0:
      sel_idx = np.arange(len(score))[np.asarray(score) < 0.0]
    score_before_check = np.asarray(score_before_check)
    sel_true_false_with_labels = score_before_check < min(max_score, 0.0)

  if args.strategy > 1:
    # check labels
    true_label = np.asarray(true_label)
    expected_label = np.asarray(expected_label)

  sel_org_idx = np.asarray(idx)[sel_idx].tolist()  # samples that are used in training
  sel_org_idx_with_labels = np.asarray(idx)[sel_true_false_with_labels].tolist() # samples that have labels
  print('calculating influence -- done')
  return sel_org_idx, sel_org_idx_with_labels


def sample_by_infl_without_true_label(args, state, val_data, unlabeled_data, num):
  """
  Get influence score of each unlabeled_data on val_data, then sample according to scores
  For fairness, the sign is very important
  """
  preprocess_func_torch2jax = gen_preprocess_func_torch2jax(args.dataset)
  if args.aux_data is None:
    preprocess_func_torch2jax_aux = gen_preprocess_func_torch2jax(args.dataset)
  else:
    preprocess_func_torch2jax_aux = gen_preprocess_func_torch2jax(args.aux_data)
  print('begin calculating influence')
  num_samples = 0.0
  grad_sum = 0.0
  grad_org_sum = 0.0
  grad_fair_sum = 0.0
  for example in val_data: # Need to run on the validation dataset to avoid the negative effect of distribution shift, e.g., DP is not robust to distribution shift. For fairness, val data may be iid as test data 
    batch = preprocess_func_torch2jax(example, args)
    bsz = batch['feature'].shape[0]
    grads, grads_org = np.asarray(infl_step(state, batch))

    grads_fair_batch = np.asarray(infl_step_fair(state, batch))

    # sum & average
    grad_sum += grads * bsz
    grad_org_sum += grads_org * bsz
    grad_fair_sum += grads_fair_batch * bsz
    num_samples += bsz

  grad_avg = (grad_sum/num_samples).reshape(-1,1)
  grad_org_avg = (grad_org_sum/num_samples).reshape(-1,1)
  grad_fair = (grad_fair_sum/num_samples).reshape(-1,1)

  # check unlabeled data
  score = []
  score_before_check = []
  idx = []
  idx_ans_pseudo_label = []
  expected_label = []
  true_label = []
  for example in unlabeled_data:
    batch = preprocess_func_torch2jax_aux(example, args)
    batch_unlabeled = batch.copy()
    batch_unlabeled['label'] = None # get grad for each label. We do not know labels of samples in unlabeled data
    grads_each_sample, logits = infl_step_per_sample(state, batch_unlabeled)
    grads_each_sample = np.asarray(grads_each_sample)
    logits = np.asarray(logits)
    infl = - np.matmul(grads_each_sample, grad_avg) # new_loss - cur_los  # 
    infl_org = - np.matmul(grads_each_sample, grad_org_avg) # new_loss - cur_los  # 
    infl_fair = - np.matmul(grads_each_sample, grad_fair)


    # Strategy 1 (baseline): random
    if args.strategy == 1:
      score += [1] * batch['label'].shape[0]
      label_expected = np.random.choice(range(args.num_classes), batch['label'].shape[0])
    # Strategy 2 (idea 1): find the label with least absolute influence, then find the sample with most negative fairness infl
    elif args.strategy == 2:
      label_expected = np.argmin(abs(infl), 1).reshape(-1)

    # Strategy 3 (idea 2): find the label with minimal influence values (most negative), then find the sample with most negative infl 
    elif args.strategy == 3:
      label_expected = np.argmin(infl, 1).reshape(-1)

    # Strategy 4: use true label
    elif args.strategy == 4:
      label_expected = batch['label'].reshape(-1)

    # Strategy 4: use model predicted label
    elif args.strategy == 5:
      label_expected = np.argmax(logits, 1).reshape(-1)
    
    # Strategy 6: baseline 1 - BALD
    elif args.strategy == 6:
      label_expected = np.argmax(logits, 1).reshape(-1)
    
    ##skip strategy 7 JTT baseline here since JTT does not depends on influence scores

    # Strategy 8: Influence selection for active learning (ISAL), select unlabeled samples according to influence scores
    elif args.strategy == 8:
      # use model predicted label as pseuod-labels
      label_expected = np.argmax(logits, 1).reshape(-1)
      
    ########################################################
    ################# sample selection #####################
    ########################################################
    
    if args.strategy > 1 and args.strategy < 6:
      score_tmp = (infl_fair[range(infl_fair.shape[0]), label_expected]).reshape(-1)

      if args.remove_pos:
        infl_expected = infl[range(infl.shape[0]), label_expected].reshape(-1)
      if args.remove_posOrg:
        infl_expected = infl_org[range(infl_org.shape[0]), label_expected].reshape(-1)
    
      if args.remove_pos or args.remove_posOrg:
        score_tmp[infl_expected > 0] = 0.0

      score += score_tmp.tolist()
      
      expected_label += label_expected.tolist()

    elif args.strategy == 6:

      score_tmp = -1 * compute_bald_score(args, softmax(logits))
      score += score_tmp.tolist()
      expected_label += label_expected.tolist()

    elif args.strategy == 8:
      infl = - np.matmul(grads_each_sample, args.H_v) # new_loss - cur_los  # 
      infl_org = - np.matmul(grads_each_sample, args.H_v_org) # new_loss - cur_los  #

      score_tmp = (infl[range(infl.shape[0]), label_expected].reshape(-1))
      score += score_tmp.tolist()
      expected_label += label_expected.tolist()

    ########################################################
    ################# sample selection #####################
    ########################################################

    idx += batch['index'].tolist()
    idx_ans_pseudo_label += [(batch['index'][i], label_expected[i]) for i in range(len(batch['index']))]

    if len(score) >= num * 100: 
      break

  if args.strategy == 1:
    sel_idx = list(range(len(score)))
    random.Random(args.infl_random_seed).shuffle(sel_idx)
    sel_idx = sel_idx[:num]

  # Strategy 2--8
  else:
    sel_idx = np.argsort(score)[:num]
    max_score = score[sel_idx[-1]]
    if max_score >= 0.0:
      sel_idx = np.arange(len(score))[np.asarray(score) < 0.0]
  
  sel_org_id = np.asarray(idx)[sel_idx].tolist()  # samples that are used in training
  sel_org_id_and_pseudo_label = np.asarray(idx_ans_pseudo_label)[sel_idx].tolist()  # samples that are used in training

  print('Finished calculating influence!')
  new_labels = {}
  for pair_i in sel_org_id_and_pseudo_label:
    new_labels[pair_i[0]] = pair_i[1]
  return sel_org_id, new_labels

def test(args, state, data):
  """
  Test
  """
  logits, labels, groups = [], [], []
  preprocess_func_torch2jax = gen_preprocess_func_torch2jax(args.dataset)
  for example in data:
    batch = preprocess_func_torch2jax(example, args, noisy_attribute = None)
    logit= test_step(state, batch)
    logits.append(logit)
    labels.append(batch['label'])
    groups.append(batch['group'])

  return compute_metrics_fair(
    logits=jnp.concatenate(logits),
    labels=jnp.concatenate(labels),
    groups=jnp.concatenate(groups),
  )

def train_celeba(args):
  # setup
  set_global_seed(args.train_seed)
  new_labels = {}
  train_loader_labeled, train_loader_unlabeled, idx_with_labels = load_data(args, args.dataset, mode = 'train', aux_dataset=args.aux_data)
  _, train_loader_unlabeled_org, idx_with_labels_org = load_data(args, args.dataset, mode = 'train', aux_dataset=None)
  args.train_with_org = True
  train_loader_new = None
  train_loader_new_org = None

  val_loader, test_loader = load_data(args, args.dataset, mode = 'val')


  preprocess_func_torch2jax = gen_preprocess_func_torch2jax(args.dataset)

  if args.aux_data is None:
    preprocess_func_torch2jax_aux = gen_preprocess_func_torch2jax(args.dataset)
  else:
    preprocess_func_torch2jax_aux = gen_preprocess_func_torch2jax(args.aux_data)


  # setup
  model = get_model(args)
  state = create_train_state(model, args)


  # get model size
  flat_tree = jax.tree_util.tree_leaves(state.params)
  num_layers = len(flat_tree)
  print(f'Numer of layers {num_layers}')

  rec = init_recorder()

  # info
  time_start = time.time()
  time_now = time_start
  print('train net...')

  # begin training
  init_val_acc = 0.0
  lmd = args.lmd
  train_step = get_train_step(args.method)
  sampled_idx = []
  sampled_idx_org = []
  idx_rec = []
  used_idx = idx_with_labels.copy()
  used_idx_org = idx_with_labels_org.copy()
  print(f'train with {args.datasize} samples (with replacement) in one epoch')

  for epoch_i in range(args.num_epochs):
    args.curr_epoch = epoch_i
    if train_loader_new is not None:
      new_iter = iter(train_loader_new)
    if train_loader_new_org is not None:
      new_iter_org = iter(train_loader_new_org)

    t = 0
    num_sample_cur = 0
    print(f'Epoch {epoch_i}')

    train_step = get_train_step(args.method)  

    ## data_loader with batch size
    while t * args.train_batch_size < args.datasize:
      
      for example in train_loader_labeled:
        new_data = 0
        if train_loader_new is not None:
          if 0 <= args.new_prob and args.new_prob <= 1 and len(train_loader_new) >= 2: # args.new_prob should be large, e.g., 0.9.   len(train_loader_new) >= len(train_loader_labeled) / 3 means the new data should be sufficient > 25 % of total
            if args.train_with_org:
              new_data = np.random.choice(range(3), p = [1.0 - args.new_prob, args.new_prob * (1.0 - args.ratio_org), args.new_prob * args.ratio_org])
            else:
              new_data = np.random.choice(range(2), p = [1.0 - args.new_prob, args.new_prob])
          else:
            new_prob = (len(train_loader_new) + 1) / (len(train_loader_new) + len(train_loader_labeled))
            if args.train_with_org:
              new_data = np.random.choice(range(3), p = [1.0 - new_prob, new_prob * (1.0 - args.ratio_org), new_prob * args.ratio_org])
            else:
              new_data = np.random.choice(range(2), p = [1.0 - new_prob, new_prob])


          if new_data == 1:
            try:
                # Samples the batch
                example = next(new_iter)
            except StopIteration:
                # restart the generator if the previous generator is exhausted.
                new_iter = iter(train_loader_new)
                example = next(new_iter)
          elif new_data == 2:
            try:
                # Samples the batch
                example = next(new_iter_org)
            except StopIteration:
                # restart the generator if the previous generator is exhausted.
                new_iter_org = iter(train_loader_new_org)
                example = next(new_iter_org)


        bsz = example[0].shape[0]

        num_sample_cur += bsz


        ###proprocess t-th batch data: example
        if new_data in [0, 2]:
          example = preprocess_func_torch2jax(example, args)
        else: 
          example = preprocess_func_torch2jax_aux(example, args, new_labels = new_labels)

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
          try:
            state, train_metric = train_step(state, batch)
          except:

            print(batch)
        else:
          raise NameError('Undefined optimization mechanism')

        rec = record_train_stats(rec, t-1, train_metric, 0)
      
        if t % args.log_steps == 0 or (t+1) * args.train_batch_size > args.datasize:

          
          test_metric = test(args, state, test_loader)
          rec, time_now = record_test(rec, t+args.datasize*epoch_i//args.train_batch_size, args.datasize*args.num_epochs//args.train_batch_size, time_now, time_start, train_metric, test_metric, metric = args.metric, warm = epoch_i < args.warm_epoch)

          val_metric = test(args, state, val_loader)
          _, time_now = record_test(rec, t+args.datasize*epoch_i//args.train_batch_size, args.datasize*args.num_epochs//args.train_batch_size, time_now, time_start, train_metric, test_metric, val_metric=val_metric, metric = args.metric, warm = epoch_i < args.warm_epoch)

          if epoch_i >= args.warm_epoch:
            if init_val_acc == 0.0:
              init_val_acc = val_metric['accuracy']
              
            '''
            Hessian approximation calculation for strategy 8
            '''
            if args.strategy == 8:
              print('start hessian approximation!')
              args.H_v, args.H_v_org = compute_hessian(state, train_loader_labeled, val_loader)

            # infl 
            args.infl_random_seed = t+args.datasize*epoch_i//args.train_batch_size + args.train_seed

            if args.aux_data == 'imagenet':
               
              sampled_idx_tmp, new_labels_tmp = sample_by_infl_without_true_label(args, state, val_loader, train_loader_unlabeled, num = args.new_data_each_round)
              new_labels.update(new_labels_tmp)
              print(f'length of new labels {len(new_labels)}')
              sampled_idx += sampled_idx_tmp
              print(f'Use {len(used_idx)}+{len(new_labels)} = {len(used_idx) + len(new_labels)} samples.')

            elif (epoch_i >= args.num_epochs - 2 and val_metric['accuracy'] >= init_val_acc - args.tol): # last two epochs
              if args.strategy == 7:
                
                print('Baseline JTT: sample misclassified examples')
                sampled_idx_tmp, new_labels_tmp = jtt_misclassified_examples(args, model, state, train_loader_labeled, num=args.new_data_each_round)

              else:
                
                sampled_idx_tmp, new_labels_tmp = sample_by_infl_without_true_label(args, state, val_loader, train_loader_unlabeled, num = args.new_data_each_round)
              new_labels.update(new_labels_tmp)
              print(f'length of new labels {len(new_labels)}')
              sampled_idx += sampled_idx_tmp
              used_idx.update(sampled_idx)
              print(f'Use {len(used_idx) - len(new_labels)}+{len(new_labels)} = {len(used_idx)} samples.')
              

            else:
              if args.strategy == 7:
                
                print('Baseline JTT: sample misclassified examples')
                sampled_idx_tmp, sel_org_idx_with_labels = jtt_misclassified_examples(args, model, state, train_loader_labeled, num=args.new_data_each_round)

              else:
                print('Sampling by influence function!')
                sampled_idx_tmp, sel_org_idx_with_labels = sample_by_infl(args, state, val_loader, train_loader_unlabeled, num = args.new_data_each_round)
              idx_with_labels.update(sel_org_idx_with_labels)
              sampled_idx += sampled_idx_tmp
              used_idx.update(sampled_idx)
              print(f'Use {len(used_idx)} samples. Get {len(idx_with_labels)} labels. Ratio: {len(used_idx)/len(idx_with_labels)}')
              idx_rec.append((epoch_i, args.infl_random_seed, used_idx, idx_with_labels))
              
            _, train_loader_unlabeled, train_loader_new, _ = load_data(args, args.dataset, mode = 'train', sampled_idx=sampled_idx, aux_dataset=args.aux_data)
            

            new_iter = iter(train_loader_new)


            if args.train_with_org:
              if args.strategy == 7:
                
                print('Baseline JTT: sample misclassified examples')
                sampled_idx_tmp, sel_org_idx_with_labels = jtt_misclassified_examples(args, model, state, train_loader_labeled, num=args.new_data_each_round)

              else:
                print('args.train_with_org = True!')
                sampled_idx_tmp, sel_org_idx_with_labels = sample_by_infl(args, state, val_loader, train_loader_unlabeled_org, num = args.new_data_each_round, force_org = args.train_with_org)
              
              idx_with_labels_org.update(sel_org_idx_with_labels)
              sampled_idx_org += sampled_idx_tmp
              used_idx_org.update(sampled_idx)
              print(f'[ADD ORG DATA] Use {len(used_idx_org)} samples. Get {len(idx_with_labels_org)} labels. Ratio: {len(used_idx_org)/len(idx_with_labels_org)}')
              _, train_loader_unlabeled_org, train_loader_new_org, _ = load_data(args, args.dataset, mode = 'train', sampled_idx=sampled_idx_org, aux_dataset=None)
            
              
              new_iter_org = iter(train_loader_new_org)


    rec = save_checkpoint(args.save_dir, t+args.datasize*epoch_i//args.train_batch_size, state, rec, save=args.save_model)


def train_compas(args):
  # setup
  set_global_seed(args.train_seed)
  new_labels = {}
  train_loader_labeled, train_loader_unlabeled, idx_with_labels = load_data(args, args.dataset, mode = 'train', aux_dataset=args.aux_data)
  _, train_loader_unlabeled_org, idx_with_labels_org = load_data(args, args.dataset, mode = 'train', aux_dataset=None)
  args.train_with_org = True
  train_loader_new = None
  train_loader_new_org = None

  val_loader, test_loader = load_data(args, args.dataset, mode = 'val')


  preprocess_func_torch2jax = gen_preprocess_func_torch2jax(args.dataset)

  if args.aux_data is None:
    preprocess_func_torch2jax_aux = gen_preprocess_func_torch2jax(args.dataset)
  else:
    preprocess_func_torch2jax_aux = gen_preprocess_func_torch2jax(args.aux_data)


  model = get_model(args)
  state = create_train_state(model, args)


  # get model size
  flat_tree = jax.tree_util.tree_leaves(state.params)
  num_layers = len(flat_tree)
  print(f'Numer of layers {num_layers}')

  rec = init_recorder()

  

  # info
  time_start = time.time()
  time_now = time_start
  print('train net...')

  # begin training
  init_val_acc = 0.0
  lmd = args.lmd
  train_step = get_train_step(args.method)
  sampled_idx = []
  sampled_idx_org = []
  idx_rec = []
  used_idx = idx_with_labels.copy()
  used_idx_org = idx_with_labels_org.copy()
  print(f'train with {args.datasize} samples (with replacement) in one epoch')

  for epoch_i in range(args.num_epochs):
    args.curr_epoch = epoch_i

    if train_loader_new is not None:
      new_iter = iter(train_loader_new)
    if train_loader_new_org is not None:
      new_iter_org = iter(train_loader_new_org)

    t = 0
    num_sample_cur = 0
    print(f'Epoch {epoch_i}')
    
    train_step = get_train_step(args.method)

  
    while t * args.train_batch_size < args.datasize:
      
      for example in train_loader_labeled:
        new_data = 0
        if train_loader_new is not None:
          if 0 <= args.new_prob and args.new_prob <= 1 and len(train_loader_new) >= 2: # args.new_prob should be large, e.g., 0.9.   len(train_loader_new) >= len(train_loader_labeled) / 3 means the new data should be sufficient > 25 % of total
            if args.train_with_org:
              new_data = np.random.choice(range(3), p = [1.0 - args.new_prob, args.new_prob * (1.0 - args.ratio_org), args.new_prob * args.ratio_org])
            else:
              new_data = np.random.choice(range(2), p = [1.0 - args.new_prob, args.new_prob])
          else:
            new_prob = (len(train_loader_new) + 1) / (len(train_loader_new) + len(train_loader_labeled))
            if args.train_with_org:
              new_data = np.random.choice(range(3), p = [1.0 - new_prob, new_prob * (1.0 - args.ratio_org), new_prob * args.ratio_org])
            else:
              new_data = np.random.choice(range(2), p = [1.0 - new_prob, new_prob])


          if new_data == 1:
            try:
                # Samples the batch
                example = next(new_iter)
            except StopIteration:
                # restart the generator if the previous generator is exhausted.
                new_iter = iter(train_loader_new)
                example = next(new_iter)
          elif new_data == 2:
            try:
                # Samples the batch
                example = next(new_iter_org)
            except StopIteration:
                # restart the generator if the previous generator is exhausted.
                new_iter_org = iter(train_loader_new_org)
                example = next(new_iter_org)


        bsz = example[0].shape[0]

        num_sample_cur += bsz

        example = preprocess_func_torch2jax(example, args)

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
          try:
            state, train_metric = train_step(state, batch)
          except:
            print(batch)
        else:
          raise NameError('Undefined optimization mechanism')

        rec = record_train_stats(rec, t-1, train_metric, 0)

        if t % args.log_steps == 0 or (t+1) * args.train_batch_size > args.datasize:
          
          test_metric = test(args, state, test_loader)
          rec, time_now = record_test(rec, t+args.datasize*epoch_i//args.train_batch_size, args.datasize*args.num_epochs//args.train_batch_size, time_now, time_start, train_metric, test_metric, metric = args.metric, warm = epoch_i < args.warm_epoch)

          val_metric = test(args, state, val_loader)
          _, time_now = record_test(rec, t+args.datasize*epoch_i//args.train_batch_size, args.datasize*args.num_epochs//args.train_batch_size, time_now, time_start, train_metric, test_metric, val_metric=val_metric, metric = args.metric, warm = epoch_i < args.warm_epoch)

          if epoch_i >= args.warm_epoch:
            if init_val_acc == 0.0:
              init_val_acc = val_metric['accuracy']
            
            '''
            Hessian approximation calculation for strategy 8
            '''
            if args.strategy == 8:
              print('start hessian approximation!')
              args.H_v, args.H_v_org = compute_hessian(state, train_loader_labeled, val_loader)
            

            # infl 
            args.infl_random_seed = t+args.datasize*epoch_i//args.train_batch_size + args.train_seed

            if (epoch_i >= args.num_epochs - 2 and val_metric['accuracy'] >= init_val_acc - args.tol): # last two epochs
              if args.strategy == 7:
                
                print('Baseline JTT: sample misclassified examples')
                sampled_idx_tmp, new_labels_tmp = jtt_misclassified_examples(args, model, state, train_loader_labeled, num=args.new_data_each_round)

              else:
                
                sampled_idx_tmp, new_labels_tmp = sample_by_infl_without_true_label(args, state, val_loader, train_loader_unlabeled, num = args.new_data_each_round)
              new_labels.update(new_labels_tmp)
              print(f'length of new labels {len(new_labels)}')
              sampled_idx += sampled_idx_tmp
              used_idx.update(sampled_idx)
              print(f'Use {len(used_idx) - len(new_labels)}+{len(new_labels)} = {len(used_idx)} samples.')
              

            else:
              if args.strategy == 7:
                
                print('Baseline JTT: sample misclassified examples')
                sampled_idx_tmp, sel_org_idx_with_labels = jtt_misclassified_examples(args, model, state, train_loader_labeled, num=args.new_data_each_round)
              else:
                print('Sampling by influence function!')
                sampled_idx_tmp, sel_org_idx_with_labels = sample_by_infl(args, state, val_loader, train_loader_unlabeled, num = args.new_data_each_round)
              idx_with_labels.update(sel_org_idx_with_labels)
              sampled_idx += sampled_idx_tmp
              used_idx.update(sampled_idx)
              print(f'Use {len(used_idx)} samples. Get {len(idx_with_labels)} labels. Ratio: {len(used_idx)/len(idx_with_labels)}')
              idx_rec.append((epoch_i, args.infl_random_seed, used_idx, idx_with_labels))

            
            train_loader_unlabeled, train_loader_new, _ = load_data(args, args.dataset, mode = 'train', sampled_idx=sampled_idx, aux_dataset=args.aux_data)

            new_iter = iter(train_loader_new)


            if args.train_with_org:
              if args.strategy == 7:
                
                sampled_idx_tmp, sel_org_idx_with_labels = jtt_misclassified_examples(args, model, state, train_loader_labeled, num=args.new_data_each_round)
              else:
                print('args.train_with_org = True!')
                sampled_idx_tmp, sel_org_idx_with_labels = sample_by_infl(args, state, val_loader, train_loader_unlabeled_org, num = args.new_data_each_round, force_org = args.train_with_org)
              idx_with_labels_org.update(sel_org_idx_with_labels)
              sampled_idx_org += sampled_idx_tmp
              used_idx_org.update(sampled_idx)
              print(f'[ADD ORG DATA] Use {len(used_idx_org)} samples. Get {len(idx_with_labels_org)} labels. Ratio: {len(used_idx_org)/len(idx_with_labels_org)}')
              
              train_loader_unlabeled_org, train_loader_new_org, _ = load_data(args, args.dataset, mode = 'train', sampled_idx=sampled_idx, aux_dataset=args.aux_data)
              
              new_iter_org = iter(train_loader_new_org)

    rec = save_checkpoint(args.save_dir, t+args.datasize*epoch_i//args.train_batch_size, state, rec, save=args.save_model)

def train_adult(args):
  # setup
  set_global_seed(args.train_seed)
  new_labels = {}
  train_loader_labeled, train_loader_unlabeled, idx_with_labels = load_data(args, args.dataset, mode = 'train', aux_dataset=args.aux_data)
  _, train_loader_unlabeled_org, idx_with_labels_org = load_data(args, args.dataset, mode = 'train', aux_dataset=None)
  args.train_with_org = True
  train_loader_new = None
  train_loader_new_org = None

  val_loader, test_loader = load_data(args, args.dataset, mode = 'val')


  preprocess_func_torch2jax = gen_preprocess_func_torch2jax(args.dataset)

  if args.aux_data is None:
    preprocess_func_torch2jax_aux = gen_preprocess_func_torch2jax(args.dataset)
  else:
    preprocess_func_torch2jax_aux = gen_preprocess_func_torch2jax(args.aux_data)


  # setup
  model = get_model(args)
  state = create_train_state(model, args)


  # get model size
  flat_tree = jax.tree_util.tree_leaves(state.params)
  num_layers = len(flat_tree)
  print(f'Numer of layers {num_layers}')

  rec = init_recorder()
  # info
  time_start = time.time()
  time_now = time_start
  print('train net...')

  # begin training
  init_val_acc = 0.0
  lmd = args.lmd
  train_step = get_train_step(args.method)
  sampled_idx = []
  sampled_idx_org = []
  idx_rec = []
  used_idx = idx_with_labels.copy()
  used_idx_org = idx_with_labels_org.copy()
  print(f'train with {args.datasize} samples (with replacement) in one epoch')

  for epoch_i in range(args.num_epochs):
    args.curr_epoch = epoch_i
    if train_loader_new is not None:
      new_iter = iter(train_loader_new)
    if train_loader_new_org is not None:
      new_iter_org = iter(train_loader_new_org)
      
    t = 0
    num_sample_cur = 0
    print(f'Epoch {epoch_i}')

    train_step = get_train_step(args.method)

    ## data_loader with batch size
    while t * args.train_batch_size < args.datasize:
      
      for example in train_loader_labeled:
        new_data = 0
        if train_loader_new is not None:
          if 0 <= args.new_prob and args.new_prob <= 1 and len(train_loader_new) >= 2: # args.new_prob should be large, e.g., 0.9.   len(train_loader_new) >= len(train_loader_labeled) / 3 means the new data should be sufficient > 25 % of total
            if args.train_with_org:
              new_data = np.random.choice(range(3), p = [1.0 - args.new_prob, args.new_prob * (1.0 - args.ratio_org), args.new_prob * args.ratio_org])
            else:
              new_data = np.random.choice(range(2), p = [1.0 - args.new_prob, args.new_prob])
          else:
            new_prob = (len(train_loader_new) + 1) / (len(train_loader_new) + len(train_loader_labeled))
            if args.train_with_org:
              new_data = np.random.choice(range(3), p = [1.0 - new_prob, new_prob * (1.0 - args.ratio_org), new_prob * args.ratio_org])
            else:
              new_data = np.random.choice(range(2), p = [1.0 - new_prob, new_prob])


          if new_data == 1:
            try:
                # Samples the batch
                example = next(new_iter)
            except StopIteration:
                # restart the generator if the previous generator is exhausted.
                new_iter = iter(train_loader_new)
                example = next(new_iter)
          elif new_data == 2:
            try:
                # Samples the batch
                example = next(new_iter_org)
            except StopIteration:
                # restart the generator if the previous generator is exhausted.
                new_iter_org = iter(train_loader_new_org)
                example = next(new_iter_org)


        bsz = example[0].shape[0]

        num_sample_cur += bsz

        example = preprocess_func_torch2jax(example, args)

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
        else:
          raise NameError('Undefined optimization mechanism')

        rec = record_train_stats(rec, t-1, train_metric, 0)
  
        if t % args.log_steps == 0 or (t+1) * args.train_batch_size > args.datasize:
          
          test_metric = test(args, state, test_loader)
          rec, time_now = record_test(rec, t+args.datasize*epoch_i//args.train_batch_size, args.datasize*args.num_epochs//args.train_batch_size, time_now, time_start, train_metric, test_metric, metric = args.metric, warm = epoch_i < args.warm_epoch)

          val_metric = test(args, state, val_loader)
          _, time_now = record_test(rec, t+args.datasize*epoch_i//args.train_batch_size, args.datasize*args.num_epochs//args.train_batch_size, time_now, time_start, train_metric, test_metric, val_metric=val_metric, metric = args.metric, warm = epoch_i < args.warm_epoch)



          if epoch_i >= args.warm_epoch:
            if init_val_acc == 0.0:
              init_val_acc = val_metric['accuracy']
              
            '''
            Hessian approximation calculation for strategy 8
            '''
            if args.strategy == 8:
              print('start hessian approximation!')
              args.H_v, args.H_v_org = compute_hessian(state, train_loader_labeled, val_loader) 

            # infl 
            args.infl_random_seed = t+args.datasize*epoch_i//args.train_batch_size + args.train_seed

            if args.aux_data == 'imagenet':
               
              sampled_idx_tmp, new_labels_tmp = sample_by_infl_without_true_label(args, state, val_loader, train_loader_unlabeled, num = args.new_data_each_round)
              new_labels.update(new_labels_tmp)
              print(f'length of new labels {len(new_labels)}')
              sampled_idx += sampled_idx_tmp
              print(f'Use {len(used_idx)}+{len(new_labels)} = {len(used_idx) + len(new_labels)} samples.')

            if (epoch_i >= args.num_epochs - 2 and val_metric['accuracy'] >= init_val_acc - args.tol): # last two epochs
              if args.strategy == 7:
                print('Baseline JTT: sample misclassified examples')
                sampled_idx_tmp, new_labels_tmp = jtt_misclassified_examples(args, model, state, train_loader_labeled, num=args.new_data_each_round)

              else:
                
                sampled_idx_tmp, new_labels_tmp = sample_by_infl_without_true_label(args, state, val_loader, train_loader_unlabeled, num = args.new_data_each_round)
              new_labels.update(new_labels_tmp)
              print(f'length of new labels {len(new_labels)}')
              sampled_idx += sampled_idx_tmp
              used_idx.update(sampled_idx)
              print(f'Use {len(used_idx) - len(new_labels)}+{len(new_labels)} = {len(used_idx)} samples.')
              

            else:
              if args.strategy == 7:
                print('Baseline JTT: sample misclassified examples')
                sampled_idx_tmp, sel_org_idx_with_labels = jtt_misclassified_examples(args, model, state, train_loader_labeled, num=args.new_data_each_round)
              else:
                print('Sampling by influence function!')
                sampled_idx_tmp, sel_org_idx_with_labels = sample_by_infl(args, state, val_loader, train_loader_unlabeled, num = args.new_data_each_round)
              idx_with_labels.update(sel_org_idx_with_labels)
              sampled_idx += sampled_idx_tmp
              used_idx.update(sampled_idx)
              print(f'Use {len(used_idx)} samples. Get {len(idx_with_labels)} labels. Ratio: {len(used_idx)/len(idx_with_labels)}')
              idx_rec.append((epoch_i, args.infl_random_seed, used_idx, idx_with_labels))

            
            train_loader_unlabeled, train_loader_new, _ = load_data(args, args.dataset, mode = 'train', sampled_idx=sampled_idx, aux_dataset=args.aux_data)

            new_iter = iter(train_loader_new)


            if args.train_with_org:
              if args.strategy == 7:
                sampled_idx_tmp, sel_org_idx_with_labels = jtt_misclassified_examples(args, model, state, train_loader_labeled, num=args.new_data_each_round)
              else:
                print('args.train_with_org = True!')
                sampled_idx_tmp, sel_org_idx_with_labels = sample_by_infl(args, state, val_loader, train_loader_unlabeled_org, num = args.new_data_each_round, force_org = args.train_with_org)
              idx_with_labels_org.update(sel_org_idx_with_labels)
              sampled_idx_org += sampled_idx_tmp
              used_idx_org.update(sampled_idx)
              print(f'[ADD ORG DATA] Use {len(used_idx_org)} samples. Get {len(idx_with_labels_org)} labels. Ratio: {len(used_idx_org)/len(idx_with_labels_org)}')
              
              train_loader_unlabeled_org, train_loader_new_org, _ = load_data(args, args.dataset, mode = 'train', sampled_idx=sampled_idx, aux_dataset=args.aux_data)
  
              new_iter_org = iter(train_loader_new_org)

    rec = save_checkpoint(args.save_dir, t+args.datasize*epoch_i//args.train_batch_size, state, rec, save=args.save_model)

def train_jigsaw(args):
  # setup
  set_global_seed(args.train_seed)
  # make_dirs(args)
  new_labels = {}
  train_loader_labeled, train_loader_unlabeled, idx_with_labels = load_data(args, args.dataset, mode = 'train', aux_dataset=args.aux_data)
  _, train_loader_unlabeled_org, idx_with_labels_org = load_data(args, args.dataset, mode = 'train', aux_dataset=None)
  args.train_with_org = True
  train_loader_new = None
  train_loader_new_org = None

  val_loader, test_loader = load_data(args, args.dataset, mode = 'val')


  preprocess_func_torch2jax = gen_preprocess_func_torch2jax(args.dataset)

  if args.aux_data is None:
    preprocess_func_torch2jax_aux = gen_preprocess_func_torch2jax(args.dataset)
  else:
    preprocess_func_torch2jax_aux = gen_preprocess_func_torch2jax(args.aux_data)


  # setup
  model = get_model(args)
  state = create_train_state(model, args)


  # get model size
  flat_tree = jax.tree_util.tree_leaves(state.params)
  num_layers = len(flat_tree)
  print(f'Numer of layers {num_layers}')

  rec = init_recorder()

  

  # info
  time_start = time.time()
  time_now = time_start
  print('train net...')

  # begin training
  init_val_acc = 0.0
  lmd = args.lmd
  train_step = get_train_step(args.method)
  # epoch_pre = 0
  sampled_idx = []
  sampled_idx_org = []
  idx_rec = []
  used_idx = idx_with_labels.copy()
  used_idx_org = idx_with_labels_org.copy()
  print(f'train with {args.datasize} samples (with replacement) in one epoch')

  for epoch_i in range(args.num_epochs):
    args.curr_epoch = epoch_i
    if train_loader_new is not None:
      new_iter = iter(train_loader_new)
    if train_loader_new_org is not None:
      new_iter_org = iter(train_loader_new_org)

    t = 0
    num_sample_cur = 0
    print(f'Epoch {epoch_i}')


    train_step = get_train_step(args.method)


    ## data_loader with batch size
    while t * args.train_batch_size < args.datasize:
      
      for example in train_loader_labeled:
        new_data = 0
        if train_loader_new is not None:
          if 0 <= args.new_prob and args.new_prob <= 1 and len(train_loader_new) >= 2: # args.new_prob should be large, e.g., 0.9.   len(train_loader_new) >= len(train_loader_labeled) / 3 means the new data should be sufficient > 25 % of total
            if args.train_with_org:
              new_data = np.random.choice(range(3), p = [1.0 - args.new_prob, args.new_prob * (1.0 - args.ratio_org), args.new_prob * args.ratio_org])
            else:
              new_data = np.random.choice(range(2), p = [1.0 - args.new_prob, args.new_prob])
          else:
            new_prob = (len(train_loader_new) + 1) / (len(train_loader_new) + len(train_loader_labeled))
            if args.train_with_org:
              new_data = np.random.choice(range(3), p = [1.0 - new_prob, new_prob * (1.0 - args.ratio_org), new_prob * args.ratio_org])
            else:
              new_data = np.random.choice(range(2), p = [1.0 - new_prob, new_prob])


          if new_data == 1:
            try:
                # Samples the batch
                example = next(new_iter)
            except StopIteration:
                # restart the generator if the previous generator is exhausted.
                new_iter = iter(train_loader_new)
                example = next(new_iter)
          elif new_data == 2:
            try:
                # Samples the batch
                example = next(new_iter_org)
            except StopIteration:
                # restart the generator if the previous generator is exhausted.
                new_iter_org = iter(train_loader_new_org)
                example = next(new_iter_org)


        bsz = example[0].shape[0]

        num_sample_cur += bsz
        
        example = preprocess_func_torch2jax(example, args)

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
      

        if t % args.log_steps == 0 or (t+1) * args.train_batch_size > args.datasize:
          
          test_metric = test(args, state, test_loader)
          rec, time_now = record_test(rec, t+args.datasize*epoch_i//args.train_batch_size, args.datasize*args.num_epochs//args.train_batch_size, time_now, time_start, train_metric, test_metric, metric = args.metric, warm = epoch_i < args.warm_epoch)

          val_metric = test(args, state, val_loader)
          _, time_now = record_test(rec, t+args.datasize*epoch_i//args.train_batch_size, args.datasize*args.num_epochs//args.train_batch_size, time_now, time_start, train_metric, test_metric, val_metric=val_metric, metric = args.metric, warm = epoch_i < args.warm_epoch)

          if epoch_i >= args.warm_epoch:
            if init_val_acc == 0.0:
              init_val_acc = val_metric['accuracy']
              
            '''
            Hessian approximation calculation for strategy 8
            '''
            if args.strategy == 8:
              print('start hessian approximation!')
              args.H_v, args.H_v_org = compute_hessian(state, train_loader_labeled, val_loader)         

            # infl 
            args.infl_random_seed = t+args.datasize*epoch_i//args.train_batch_size + args.train_seed

            if args.aux_data == 'imagenet':
               
              sampled_idx_tmp, new_labels_tmp = sample_by_infl_without_true_label(args, state, val_loader, train_loader_unlabeled, num = args.new_data_each_round)
              new_labels.update(new_labels_tmp)
              print(f'length of new labels {len(new_labels)}')
              sampled_idx += sampled_idx_tmp
              print(f'Use {len(used_idx)}+{len(new_labels)} = {len(used_idx) + len(new_labels)} samples.')

            if (epoch_i >= args.num_epochs - 2 and val_metric['accuracy'] >= init_val_acc - args.tol): # last two epochs
              if args.strategy == 7:
                print('Baseline JTT: sample misclassified examples')
                sampled_idx_tmp, new_labels_tmp = jtt_misclassified_examples(args, model, state, train_loader_labeled, num=args.new_data_each_round)

              else:
                
                sampled_idx_tmp, new_labels_tmp = sample_by_infl_without_true_label(args, state, val_loader, train_loader_unlabeled, num = args.new_data_each_round)
              new_labels.update(new_labels_tmp)
              print(f'length of new labels {len(new_labels)}')
              sampled_idx += sampled_idx_tmp
              used_idx.update(sampled_idx)
              print(f'Use {len(used_idx) - len(new_labels)}+{len(new_labels)} = {len(used_idx)} samples.')
              

            else:
              if args.strategy == 7:
                print('Baseline JTT: sample misclassified examples')
                sampled_idx_tmp, sel_org_idx_with_labels = jtt_misclassified_examples(args, model, state, train_loader_labeled, num=args.new_data_each_round)
              else:
                print('Sampling by influence function!')
                sampled_idx_tmp, sel_org_idx_with_labels = sample_by_infl(args, state, val_loader, train_loader_unlabeled, num = args.new_data_each_round)
              idx_with_labels.update(sel_org_idx_with_labels)
              sampled_idx += sampled_idx_tmp
              used_idx.update(sampled_idx)
              print(f'Use {len(used_idx)} samples. Get {len(idx_with_labels)} labels. Ratio: {len(used_idx)/len(idx_with_labels)}')
              idx_rec.append((epoch_i, args.infl_random_seed, used_idx, idx_with_labels))

            
            train_loader_unlabeled, train_loader_new, _ = load_data(args, args.dataset, mode = 'train', sampled_idx=sampled_idx, aux_dataset=args.aux_data)

            new_iter = iter(train_loader_new)
            if args.train_with_org:
              if args.strategy == 7:
                sampled_idx_tmp, sel_org_idx_with_labels = jtt_misclassified_examples(args, model, state, train_loader_labeled, num=args.new_data_each_round)
              else:
                print('args.train_with_org = True!')
                sampled_idx_tmp, sel_org_idx_with_labels = sample_by_infl(args, state, val_loader, train_loader_unlabeled_org, num = args.new_data_each_round, force_org = args.train_with_org)
              idx_with_labels_org.update(sel_org_idx_with_labels)
              sampled_idx_org += sampled_idx_tmp
              used_idx_org.update(sampled_idx)
              print(f'[ADD ORG DATA] Use {len(used_idx_org)} samples. Get {len(idx_with_labels_org)} labels. Ratio: {len(used_idx_org)/len(idx_with_labels_org)}')
              
              train_loader_unlabeled_org, train_loader_new_org, _ = load_data(args, args.dataset, mode = 'train', sampled_idx=sampled_idx, aux_dataset=args.aux_data)
              
              new_iter_org = iter(train_loader_new_org)

    rec = save_checkpoint(args.save_dir, t+args.datasize*epoch_i//args.train_batch_size, state, rec, save=args.save_model)
