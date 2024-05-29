
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



def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


def lissa(matrix_fn: Callable[[Any], Any],
          vector: Any,
          recursion_depth: int,
          scale: float = 10,
          damping: float = 0.0,
          log_progress: bool = False) -> Any:
  """Estimates A^{-1}v following the LiSSA algorithm.
  See the paper http://arxiv.org/pdf/1602.03943.pdf.
  [A^{-1}v]_{j+1} = v + (I - (A + d * I))[A^{-1}v]_j * v
  Args:
    matrix_fn: Function taking a vector v and returning Av.
    vector: The vector v for which we want to compute A^{-1}v.
    recursion_depth: Depth of the LiSSA iteration.
    scale: Rescaling factor for A; the algorithm requires ||A / scale|| < 1.
    damping: Damping factor to regularize a nearly-singular A.
    log_progress: If True reports progress.
  Returns:
    An estimate of A^{-1}v.
  """
  if not damping >= 0.0:
    raise ValueError("Damping factor should be positive.")
  if not scale >= 1.0:
    raise ValueError("Scaling factor should be larger than 1.0.")

  curr_estimate = vector
  for i in range(recursion_depth):
    matrix_vector = matrix_fn(curr_estimate)
    curr_estimate = tree_map(
        lambda v, u, h: v + (1 - damping) * u - h / scale, vector,
        curr_estimate, matrix_vector)
    if log_progress:
      logging.info("LISSA: %d", i)

  curr_estimate = jax.tree_map(lambda x: x / scale, curr_estimate)
  return curr_estimate


def extract_top_n_layers_params(params, n):
    top_n_params = {}
    layer_names = list(params.keys()) 
    for layer_name in layer_names[:n]:
        top_n_params[layer_name] = params[layer_name]
    return top_n_params



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

  # train_loader_labeled, train_loader_unlabeled, idx_with_labels = load_data(args, args.dataset, mode = 'train', aux_dataset=args.aux_data)
  for example in train_loader_labeled:
    if example_size == 0:
      break
    
    batch_labeled = preprocess_func_torch2jax(example, args)
    # grads_each_sample = np.asarray(infl_step_per_sample(state, batch_unlabeled))

    # grads, grads_org = np.asarray(infl_step(state, batch_labeled))
    # grads, grads_org = infl_step(state, batch_labeled)
    grads_each_sample, logits = infl_step_per_sample(state, batch_labeled)

    # I = np.eye(grads_each_sample.shape[-1])
    #compute the gradient of labeled dataset
    for idx in range(grads_each_sample.shape[0]):
      grad_sample = grads_each_sample[idx].reshape(-1,1)

      # outer_product = np.outer(grad_sample, grad_sample)
      # dot_product = np.dot(grad_sample, H_v_tmp)

      # H_v_tmp_test = grad_avg + np.matmul(I - np.matmul(grad_sample,grad_sample), H_v_tmp)
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
    # assert predictions.ndim >= 3
    # [n_sample, n_class, ..., n_iterations]

    # expected_entropy = -np.mean(
    #     np.sum(xlogy(predictions, predictions), axis=1), axis=-1
    # )  # [batch size, ...]
    # expected_p = np.mean(predictions, axis=-1)  # [batch_size, n_classes, ...]
    # entropy_expected_p = -np.sum(xlogy(expected_p, expected_p), axis=1)  # [batch size, ...]
    # bald_acq = entropy_expected_p - expected_entropy
    # return bald_acq

    assert predictions.ndim == 2
    entropy = -np.sum(xlogy(predictions, predictions), axis=1)
    return entropy
    

def sample_strategy_7_misclassified_examples(args, model, state, train_loader, num):
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
    # grads_each_sample = np.asarray(infl_step_per_sample(state, batch_unlabeled))
    grads_each_sample, logits = infl_step_per_sample(state, batch_unlabeled)
    grads_each_sample = np.asarray(grads_each_sample)
    # grad_org_each_sample = np.asarray(grad_org_each_sample)
    logits = np.asarray(logits)

    '''
    calculate the influence of prediction/fairness component
    '''
    infl = - np.matmul(grads_each_sample, grad_avg) # new_loss - cur_los  # 
    infl_org = - np.matmul(grads_each_sample, grad_org_avg) # new_loss - cur_los  # 
    infl_fair = - np.matmul(grads_each_sample, grad_fair)


    # label_expected = np.argmin(abs(infl), 1).reshape(-1)
    # label_expected = batch['label'].reshape(-1)
    # infl_fair = (infl_fair[range(infl_fair.shape[0]), label_expected]).reshape(-1)  # assume knowing true labels 
    # tolerance = args.tol # get an unfair sample wp tolerance
    # infl_fair[infl_fair > 0] = np.random.rand(int(np.sum(infl_fair > 0))) - tolerance


    # infl_fair = (infl_fair[range(infl_fair.shape[0]), label_expected]).reshape(-1)  # use expected labels
    # infl_fair = np.asarray([-1] * infl_fair.shape[0])  # only consider acc



    # case 1: only consider fairness loss
    # case 1-1: use true labels
    # case 1-2: use model predicted labels
    # case 1-3: use min_y abs(infl)
    # case 1-4: use min_y infl
    # infl_fair = np.asarray([-1] * infl_fair.shape[0])  # only fairness. note it has been reversed

    # ----------    Reversed strategy -------------------
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
      # infl_fair_expected = infl_fair[range(infl_fair.shape[0]), label_expected.reshape(-1)].reshape(-1)
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
    # print(len(score))
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

    # score_org = np.asarray(score_org)
    # sel_true_false_with_labels = score_org <= max_score

  if args.strategy > 1:
    # check labels
    true_label = np.asarray(true_label)
    expected_label = np.asarray(expected_label)
    #expect_acc = np.mean(1.0 * (true_label == expected_label))
    #print(f'[Strategy {args.strategy}] Acc of expected label: {expect_acc}')  

    # print(f'[Strategy {args.strategy}] Expected label {expected_label}')  
    # print(f'[Strategy {args.strategy}] True label {true_label}')  

  sel_org_idx = np.asarray(idx)[sel_idx].tolist()  # samples that are used in training
  sel_org_idx_with_labels = np.asarray(idx)[sel_true_false_with_labels].tolist() # samples that have labels
  # pdb.set_trace()
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
    # grads_each_sample = np.asarray(infl_step_per_sample(state, batch_unlabeled))
    grads_each_sample, logits = infl_step_per_sample(state, batch_unlabeled)
    grads_each_sample = np.asarray(grads_each_sample)
    # grad_org_each_sample = np.asarray(grad_org_each_sample)
    logits = np.asarray(logits)
    # pdb.set_trace()
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
        # score_before_check += score_tmp.tolist()



      score += score_tmp.tolist()
      
      expected_label += label_expected.tolist()

    elif args.strategy == 6:

      score_tmp = -1 * compute_bald_score(args, softmax(logits))
      score += score_tmp.tolist()
      expected_label += label_expected.tolist()

    elif args.strategy == 8:
      # strategy 8: only use the prediction influence, not use fairness influence
      # print('Strategy 8: strating Hessian calculation!! ')

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

    # print(len(score))
    if len(score) >= num * 100: # 100
      break


  if args.strategy == 1:
    sel_idx = list(range(len(score)))
    random.Random(args.infl_random_seed).shuffle(sel_idx)
    sel_idx = sel_idx[:num]
    # sel_true_false_with_labels = sel_idx

  # Strategy 2--8
  else:
    sel_idx = np.argsort(score)[:num]
    max_score = score[sel_idx[-1]]
    if max_score >= 0.0:
      sel_idx = np.arange(len(score))[np.asarray(score) < 0.0]
    # score_before_check = np.asarray(score_before_check)
    # sel_true_false_with_labels = score_before_check < min(max_score, 0.0)
  
  sel_org_id = np.asarray(idx)[sel_idx].tolist()  # samples that are used in training
  sel_org_id_and_pseudo_label = np.asarray(idx_ans_pseudo_label)[sel_idx].tolist()  # samples that are used in training


  # sel_org_idx_with_labels = np.asarray(idx)[sel_true_false_with_labels].tolist() # samples that have labels
  print('calculating influence -- done')
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

def train(args):
  # setup
  set_global_seed(args.train_seed)
  # make_dirs(args)

  [train_loader_labeled, train_loader_unlabeled], part_1 = load_celeba_dataset_torch(args, shuffle_files=True, split='train', batch_size=args.train_batch_size, ratio = args.label_ratio)
  idx_with_labels = set(part_1)
  
  [val_loader, test_loader], _ = load_celeba_dataset_torch(args, shuffle_files=True, split='test', batch_size=args.test_batch_size, ratio = args.val_ratio)

  preprocess_func_torch2jax = gen_preprocess_func_torch2jax(args.dataset)

  args.image_shape = args.img_size
  # setup
  tmp_model = get_model(args)
  if len(tmp_model) == 2:
    model, model_linear = tmp_model
  else:
    model = tmp_model

  state = create_train_state(model, args)


  # get model size
  flat_tree = jax.tree_util.tree_leaves(state.params)
  num_layers = len(flat_tree)
  print(f'Numer of layers {num_layers}')

  rec = init_recorder()

  

  # info
  # log_and_save_args(args)
  time_start = time.time()
  time_now = time_start
  print('train net...')

  # begin training
  lmd = args.lmd
  loss = 0.0
  loss_rec = [loss]
  train_step = get_train_step(args.method)
  # epoch_pre = 0
  sampled_idx = []
  idx_rec = []
  for epoch_i in range(args.num_epochs):

    args.curr_epoch = epoch_i
    t = 0
    num_sample_cur = 0
    print(f'Epoch {epoch_i}')
    # if epoch_i < args.warm_epoch: 
    #   print(f'warmup epoch = {epoch_i+1}/{args.warm_epoch}')
    # if epoch_i == args.warm_epoch:
    #   state_reg = create_train_state(model, args, params=state.params) # use the full model
    while t * args.train_batch_size < args.datasize:
      for example in train_loader_labeled:

        bsz = example[0].shape[0]


        num_sample_cur += bsz
        example = preprocess_func_torch2jax(example, args, noisy_attribute = None)
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

        '''
        train using tran_step according to args.method
        '''
        #train
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
          rec, time_now = record_test(rec, t+args.datasize*epoch_i//args.train_batch_size, args.datasize*args.num_epochs//args.train_batch_size, time_now, time_start, train_metric, test_metric, metric = args.metric)
          if epoch_i >= args.warm_epoch:
            # infl 
            args.infl_random_seed = t+args.datasize*epoch_i//args.train_batch_size + args.train_seed

            #using samples selected by influence function
            sampled_idx_tmp, sel_org_idx_with_labels= sample_by_infl(args, state, val_loader, train_loader_unlabeled, num = args.new_data_each_round)
            sampled_idx += sampled_idx_tmp
            idx_with_labels.update(sel_org_idx_with_labels)
            val_metric = test(args, state, val_loader)
            _, time_now = record_test(rec, t+args.datasize*epoch_i//args.train_batch_size, args.datasize*args.num_epochs//args.train_batch_size, time_now, time_start, train_metric, test_metric, val_metric=val_metric, metric = args.metric)


            [train_loader_labeled, train_loader_unlabeled], _ = load_celeba_dataset_torch(args, shuffle_files=True, split='train', batch_size=args.train_batch_size, ratio = args.label_ratio, sampled_idx=sampled_idx)
            used_idx = set(part_1 + sampled_idx)
            print(f'Use {len(used_idx)} samples. Get {len(idx_with_labels)} labels. Ratio: {len(used_idx)/len(idx_with_labels)}')
            idx_rec.append((epoch_i, args.infl_random_seed, used_idx, idx_with_labels))
            save_name = f'./results/s{args.strategy}_{args.metric}_{args.label_ratio}_new{args.new_data_each_round}_100round.npy'
            np.save(save_name, idx_rec)

          
          # print(f'lmd is {lmd}')



    rec = save_checkpoint(args.save_dir, t+args.datasize*epoch_i//args.train_batch_size, state, rec, save=args.save_model)#save_checkpoint: save=True means save model

  # wrap it up
  # save_recorder(args.save_dir, rec)
  save_name = f'./results/s{args.strategy}_{args.metric}_{args.label_ratio}_new{args.new_data_each_round}_100round.npy'
  np.save(save_name, idx_rec)




def fair_train(args):
  # setup
  set_global_seed(args.train_seed)
  make_dirs(args)
  preprocess_func_torch2jax = gen_preprocess_func_torch2jax(args.dataset)


  if args.strategy == 1:
    [_, _], part1, part2 = load_celeba_dataset_torch(args, shuffle_files=True, split='train', batch_size=args.train_batch_size, ratio = args.label_ratio, sampled_idx=[], return_part2=True)
    load_name = f'./results/s2_{args.metric}_{args.label_ratio}_new{args.new_data_each_round}_100round_case1_remove_unfair_trainConf{args.train_conf}_posloss{args.remove_pos}_poslossOrg{args.remove_posOrg}.npy'
    indices = np.load(load_name, allow_pickle=True)
    sel_idx = list(indices[args.sel_round][2])
    num_sample_to_add = len(sel_idx) - len(part1)
    random.Random(args.train_seed).shuffle(part2)
    sel_idx = part1 + part2[:num_sample_to_add]
    print(f'randomly select {len(part1)} + {num_sample_to_add} = {len(sel_idx)} samples')
  elif args.strategy in [2,3,4,5]:
    load_name = f'./results/s{args.strategy}_{args.metric}_{args.label_ratio}_new{args.new_data_each_round}_100round_case1_remove_unfair_trainConf{args.train_conf}_posloss{args.remove_pos}_poslossOrg{args.remove_posOrg}.npy'
    indices = np.load(load_name, allow_pickle=True)
    sel_idx = list(indices[args.sel_round][2])
  elif args.strategy == 6:
    [_, _], sel_idx = load_celeba_dataset_torch(args, shuffle_files=True, split='train', batch_size=args.train_batch_size, ratio = args.label_ratio, sampled_idx=None, fair_train=True) # use all data for training
  else:
    raise NameError('We only have strategies from 1 to 6')

  [train_loader_labeled, _], _ = load_celeba_dataset_torch(args, shuffle_files=True, split='train', batch_size=args.train_batch_size, ratio = 0.0, sampled_idx=sel_idx, fair_train=True)

  
  [val_loader, test_loader], _ = load_celeba_dataset_torch(args, shuffle_files=True, split='test', batch_size=args.test_batch_size, ratio = args.val_ratio, fair_train=True)

  args.image_shape = args.img_size
  # setup
  model, model_linear = get_model(args)
  args.hidden_size = model_linear.hidden_size
  state, lr_scheduler = create_train_state(model, args, return_opt=True)


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

    while t * args.train_batch_size < args.datasize:
      for example in train_loader_labeled:

        
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
          print(f'[Step {state.step}] Conf: {args.conf} Current lr is {np.round(lr_scheduler(state.step), 5)}')
          test_metric = test(args, state, test_loader)
          val_metric = test(args, state, val_loader)
          worst_group_id = np.argmin(val_metric['acc'])
          _, time_now = record_test(rec, t+args.datasize*epoch_i//args.train_batch_size, args.datasize*args.num_epochs//args.train_batch_size, time_now, time_start, train_metric, test_metric, val_metric=val_metric)
          rec, time_now = record_test(rec, t+args.datasize*epoch_i//args.train_batch_size, args.datasize*args.num_epochs//args.train_batch_size, time_now, time_start, train_metric, test_metric)

          print(f'lmd is {lmd}')

    rec = save_checkpoint(args.save_dir, t+args.datasize*epoch_i//args.train_batch_size, state, rec, save=args.save_model) 

  # wrap it up
  file_name = f'/s{args.strategy}_{args.metric}_{args.label_ratio}_new{args.new_data_each_round}_100round_case1_remove_unfair_trainConf{args.train_conf}_posloss{args.remove_pos}_poslossOrg{args.remove_posOrg}_{args.sel_round}.pkl'
  save_recorder(args.save_dir, rec, file_name=file_name)



def train_celeba(args):
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
      
      #####################################################################
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
          #print(f'using the {args.dataset} as example')
          example = preprocess_func_torch2jax(example, args)
          
        else: #new_data =1
          #print(f'using the {args.aux_data} as example')
          example = preprocess_func_torch2jax_aux(example, args, new_labels = new_labels)

        t += 1
        if t * args.train_batch_size > args.datasize:
          break
        #####################################################################


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
          # state, train_metric = train_step(state, batch)
          try:
            state, train_metric = train_step(state, batch)
          except:

            print(batch)
        # elif args.method in ['fix_lmd','dynamic_lmd']:
        #   state, train_metric, lmd = train_step(state, batch, lmd = lmd, T=None)
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
              print('##########If: Using the aux_data: imagenet!')
              sampled_idx_tmp, new_labels_tmp = sample_by_infl_without_true_label(args, state, val_loader, train_loader_unlabeled, num = args.new_data_each_round)
              new_labels.update(new_labels_tmp)
              print(f'length of new labels {len(new_labels)}')
              sampled_idx += sampled_idx_tmp
              # used_idx.update(sampled_idx)
              print(f'Use {len(used_idx)}+{len(new_labels)} = {len(used_idx) + len(new_labels)} samples.')

            elif (epoch_i >= args.num_epochs - 2 and val_metric['accuracy'] >= init_val_acc - args.tol): # last two epochs
              if args.strategy == 7:
                ## my code here##
                print('Baseline JTT: sample misclassified examples')
                sampled_idx_tmp, new_labels_tmp = sample_strategy_7_misclassified_examples(args, model, state, train_loader_labeled, num=args.new_data_each_round)

              else:
                print('#############Elif: Using the aux_data: imagenet!')
                sampled_idx_tmp, new_labels_tmp = sample_by_infl_without_true_label(args, state, val_loader, train_loader_unlabeled, num = args.new_data_each_round)
              new_labels.update(new_labels_tmp)
              print(f'length of new labels {len(new_labels)}')
              sampled_idx += sampled_idx_tmp
              used_idx.update(sampled_idx)
              print(f'Use {len(used_idx) - len(new_labels)}+{len(new_labels)} = {len(used_idx)} samples.')
              

            else:
              if args.strategy == 7:
                ## my code here##
                print('Baseline JTT: sample misclassified examples')
                sampled_idx_tmp, sel_org_idx_with_labels = sample_strategy_7_misclassified_examples(args, model, state, train_loader_labeled, num=args.new_data_each_round)

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
                ## my code here##
                print('Baseline JTT: sample misclassified examples')
                sampled_idx_tmp, sel_org_idx_with_labels = sample_strategy_7_misclassified_examples(args, model, state, train_loader_labeled, num=args.new_data_each_round)

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

  
    while t * args.train_batch_size < args.datasize:
      
      #####################################################################
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
        #####################################################################


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
            # print(f'train_metric: {train_metric}')
          except:
            print(batch)
        else:
          raise NameError('Undefined optimization mechanism')

        rec = record_train_stats(rec, t-1, train_metric, 0)


        # # test the test metric for each batch
        # test_metric = test(args, state, test_loader)
        # rec, time_now = record_test(rec, t+args.datasize*epoch_i//args.train_batch_size, args.datasize*args.num_epochs//args.train_batch_size, time_now, time_start, train_metric, test_metric, metric = args.metric, warm = epoch_i < args.warm_epoch)

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
                ## my code here##
                print('Baseline JTT: sample misclassified examples')
                sampled_idx_tmp, new_labels_tmp = sample_strategy_7_misclassified_examples(args, model, state, train_loader_labeled, num=args.new_data_each_round)

              else:
                print('#############Elif: Using the aux_data: imagenet!')
                sampled_idx_tmp, new_labels_tmp = sample_by_infl_without_true_label(args, state, val_loader, train_loader_unlabeled, num = args.new_data_each_round)
              new_labels.update(new_labels_tmp)
              print(f'length of new labels {len(new_labels)}')
              sampled_idx += sampled_idx_tmp
              used_idx.update(sampled_idx)
              print(f'Use {len(used_idx) - len(new_labels)}+{len(new_labels)} = {len(used_idx)} samples.')
              

            else:
              if args.strategy == 7:
                ## my code here##
                print('Baseline JTT: sample misclassified examples')
                sampled_idx_tmp, sel_org_idx_with_labels = sample_strategy_7_misclassified_examples(args, model, state, train_loader_labeled, num=args.new_data_each_round)
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
                ## my code here##
                sampled_idx_tmp, sel_org_idx_with_labels = sample_strategy_7_misclassified_examples(args, model, state, train_loader_labeled, num=args.new_data_each_round)
              else:
                print('args.train_with_org = True!')
                sampled_idx_tmp, sel_org_idx_with_labels = sample_by_infl(args, state, val_loader, train_loader_unlabeled_org, num = args.new_data_each_round, force_org = args.train_with_org)
              idx_with_labels_org.update(sel_org_idx_with_labels)
              sampled_idx_org += sampled_idx_tmp
              used_idx_org.update(sampled_idx)
              print(f'[ADD ORG DATA] Use {len(used_idx_org)} samples. Get {len(idx_with_labels_org)} labels. Ratio: {len(used_idx_org)/len(idx_with_labels_org)}')
              #_, train_loader_unlabeled_org, train_loader_new_org, _ = load_data(args, args.dataset, mode = 'train', sampled_idx=sampled_idx_org, aux_dataset=None)
              
              train_loader_unlabeled_org, train_loader_new_org, _ = load_data(args, args.dataset, mode = 'train', sampled_idx=sampled_idx, aux_dataset=args.aux_data)
              
              new_iter_org = iter(train_loader_new_org)

    rec = save_checkpoint(args.save_dir, t+args.datasize*epoch_i//args.train_batch_size, state, rec, save=args.save_model)

  # wrap it up
  # save_recorder(args.save_dir, rec)
  # save_name = f'./results/s{args.strategy}_{args.metric}_{args.label_ratio}_new{args.new_data_each_round}_100round.npy'
  # np.save(save_name, idx_rec)


def train_adult(args):
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
      
      #####################################################################
      for example in train_loader_labeled:

        # import pdb
        # pdb.set_trace()
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
        # if new_data in [0, 2]:
        #   print(f'using the {args.dataset} as example')
        #   example = preprocess_func_torch2jax(example, args)
          
        # else: #new_data =1
        #   print(f'using the {args.aux_data} as example')
        #   example = preprocess_func_torch2jax_aux(example, args, new_labels = new_labels)
        
        #print('length of example: ' +str(len(example)))

        example = preprocess_func_torch2jax(example, args)

        t += 1
        if t * args.train_batch_size > args.datasize:
          break
        #####################################################################


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
          # state, train_metric = train_step(state, batch)
          # # try:
          state, train_metric = train_step(state, batch)
          # except:
          #   print(batch)
        # elif args.method in ['fix_lmd','dynamic_lmd']:
        #   state, train_metric, lmd = train_step(state, batch, lmd = lmd, T=None)
        else:
          raise NameError('Undefined optimization mechanism')

        rec = record_train_stats(rec, t-1, train_metric, 0)
    
        
        # print('####### batch:: label 1: ' + str(sum(example['label'])) + '; ##### label 0: ' + str(len(example['label']) - sum(example['label'])))

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
              print('##########If: Using the aux_data: imagenet!')
              sampled_idx_tmp, new_labels_tmp = sample_by_infl_without_true_label(args, state, val_loader, train_loader_unlabeled, num = args.new_data_each_round)
              new_labels.update(new_labels_tmp)
              print(f'length of new labels {len(new_labels)}')
              sampled_idx += sampled_idx_tmp
              # used_idx.update(sampled_idx)
              print(f'Use {len(used_idx)}+{len(new_labels)} = {len(used_idx) + len(new_labels)} samples.')

            if (epoch_i >= args.num_epochs - 2 and val_metric['accuracy'] >= init_val_acc - args.tol): # last two epochs
              if args.strategy == 7:
                ## my code here##
                print('Baseline JTT: sample misclassified examples')
                sampled_idx_tmp, new_labels_tmp = sample_strategy_7_misclassified_examples(args, model, state, train_loader_labeled, num=args.new_data_each_round)

              else:
                print('#############Elif: Using the aux_data: imagenet!')
                sampled_idx_tmp, new_labels_tmp = sample_by_infl_without_true_label(args, state, val_loader, train_loader_unlabeled, num = args.new_data_each_round)
              new_labels.update(new_labels_tmp)
              print(f'length of new labels {len(new_labels)}')
              sampled_idx += sampled_idx_tmp
              used_idx.update(sampled_idx)
              print(f'Use {len(used_idx) - len(new_labels)}+{len(new_labels)} = {len(used_idx)} samples.')
              

            else:
              if args.strategy == 7:
                ## my code here##
                print('Baseline JTT: sample misclassified examples')
                sampled_idx_tmp, sel_org_idx_with_labels = sample_strategy_7_misclassified_examples(args, model, state, train_loader_labeled, num=args.new_data_each_round)
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
                ## my code here##
                sampled_idx_tmp, sel_org_idx_with_labels = sample_strategy_7_misclassified_examples(args, model, state, train_loader_labeled, num=args.new_data_each_round)
              else:
                print('args.train_with_org = True!')
                sampled_idx_tmp, sel_org_idx_with_labels = sample_by_infl(args, state, val_loader, train_loader_unlabeled_org, num = args.new_data_each_round, force_org = args.train_with_org)
              idx_with_labels_org.update(sel_org_idx_with_labels)
              sampled_idx_org += sampled_idx_tmp
              used_idx_org.update(sampled_idx)
              print(f'[ADD ORG DATA] Use {len(used_idx_org)} samples. Get {len(idx_with_labels_org)} labels. Ratio: {len(used_idx_org)/len(idx_with_labels_org)}')
              #_, train_loader_unlabeled_org, train_loader_new_org, _ = load_data(args, args.dataset, mode = 'train', sampled_idx=sampled_idx_org, aux_dataset=None)
              
              train_loader_unlabeled_org, train_loader_new_org, _ = load_data(args, args.dataset, mode = 'train', sampled_idx=sampled_idx, aux_dataset=args.aux_data)
  
              new_iter_org = iter(train_loader_new_org)




    rec = save_checkpoint(args.save_dir, t+args.datasize*epoch_i//args.train_batch_size, state, rec, save=args.save_model)

  # wrap it up
  # save_recorder(args.save_dir, rec)
  # save_name = f'./results/s{args.strategy}_{args.metric}_{args.label_ratio}_new{args.new_data_each_round}_100round.npy'
  # np.save(save_name, idx_rec)


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
        #####################################################################


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
          # state, train_metric = train_step(state, batch)
          # try:
          #print('trying to do train step!!!')
          state, train_metric = train_step(state, batch)
          # except:
          #   print(batch)
        elif args.method in ['fix_lmd','dynamic_lmd']:
          state, train_metric, lmd = train_step(state, batch, lmd = lmd, T=None)
        else:
          raise NameError('Undefined optimization mechanism')

        rec = record_train_stats(rec, t-1, train_metric, 0)
      
        # # test the test metric for each batch
        # test_metric = test(args, state, test_loader)
        # rec, time_now = record_test(rec, t+args.datasize*epoch_i//args.train_batch_size, args.datasize*args.num_epochs//args.train_batch_size, time_now, time_start, train_metric, test_metric, metric = args.metric, warm = epoch_i < args.warm_epoch)
        # print('############batch 1: ' + str(sum(example['label'])) + '; ##### batch 0: ' + str(len(example['label']) - sum(example['label'])))

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
              print('##########If: Using the aux_data: imagenet!')
              sampled_idx_tmp, new_labels_tmp = sample_by_infl_without_true_label(args, state, val_loader, train_loader_unlabeled, num = args.new_data_each_round)
              new_labels.update(new_labels_tmp)
              print(f'length of new labels {len(new_labels)}')
              sampled_idx += sampled_idx_tmp
              # used_idx.update(sampled_idx)
              print(f'Use {len(used_idx)}+{len(new_labels)} = {len(used_idx) + len(new_labels)} samples.')

            if (epoch_i >= args.num_epochs - 2 and val_metric['accuracy'] >= init_val_acc - args.tol): # last two epochs
              if args.strategy == 7:
                ## my code here##
                print('Baseline JTT: sample misclassified examples')
                sampled_idx_tmp, new_labels_tmp = sample_strategy_7_misclassified_examples(args, model, state, train_loader_labeled, num=args.new_data_each_round)

              else:
                print('#############Elif: Using the aux_data: imagenet!')
                sampled_idx_tmp, new_labels_tmp = sample_by_infl_without_true_label(args, state, val_loader, train_loader_unlabeled, num = args.new_data_each_round)
              new_labels.update(new_labels_tmp)
              print(f'length of new labels {len(new_labels)}')
              sampled_idx += sampled_idx_tmp
              used_idx.update(sampled_idx)
              print(f'Use {len(used_idx) - len(new_labels)}+{len(new_labels)} = {len(used_idx)} samples.')
              

            else:
              if args.strategy == 7:
                ## my code here##
                print('Baseline JTT: sample misclassified examples')
                sampled_idx_tmp, sel_org_idx_with_labels = sample_strategy_7_misclassified_examples(args, model, state, train_loader_labeled, num=args.new_data_each_round)
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
                ## my code here##
                sampled_idx_tmp, sel_org_idx_with_labels = sample_strategy_7_misclassified_examples(args, model, state, train_loader_labeled, num=args.new_data_each_round)
              else:
                print('args.train_with_org = True!')
                sampled_idx_tmp, sel_org_idx_with_labels = sample_by_infl(args, state, val_loader, train_loader_unlabeled_org, num = args.new_data_each_round, force_org = args.train_with_org)
              idx_with_labels_org.update(sel_org_idx_with_labels)
              sampled_idx_org += sampled_idx_tmp
              used_idx_org.update(sampled_idx)
              print(f'[ADD ORG DATA] Use {len(used_idx_org)} samples. Get {len(idx_with_labels_org)} labels. Ratio: {len(used_idx_org)/len(idx_with_labels_org)}')
              #_, train_loader_unlabeled_org, train_loader_new_org, _ = load_data(args, args.dataset, mode = 'train', sampled_idx=sampled_idx_org, aux_dataset=None)
              
              train_loader_unlabeled_org, train_loader_new_org, _ = load_data(args, args.dataset, mode = 'train', sampled_idx=sampled_idx, aux_dataset=args.aux_data)
              
              new_iter_org = iter(train_loader_new_org)

    rec = save_checkpoint(args.save_dir, t+args.datasize*epoch_i//args.train_batch_size, state, rec, save=args.save_model)

  # wrap it up
  # save_recorder(args.save_dir, rec)
  # save_name = f'./results/s{args.strategy}_{args.metric}_{args.label_ratio}_new{args.new_data_each_round}_100round.npy'
  # np.save(save_name, idx_rec)

