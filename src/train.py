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


     
def get_T_p(config, noisy_attribute, lr = 0.1, true_attribute = None):

  # Estimate T and P with HOC
  T_est, p_est = get_T_global_min(config, noisy_attribute, lr = lr)
  print(f'\n\n-----------------------------------------')
  print(f'Estimation finished!')
  # np.set_printoptions(precision=1)
  print(f'The estimated T (*100) is \n{np.round(T_est*100,1)}')
  print(f'The estimated p (*100) is \n{np.round(p_est*100,1)}')
  if true_attribute is not None:
      T_true, p_true = check_T(KINDS=config.num_classes, clean_label=true_attribute, noisy_label=noisy_attribute)
      # print(f'T_inv: \nest: \n{np.linalg.inv(T_est)}\ntrue:\n{np.linalg.inv(T_true)}')
      print(f'T_true: {T_true},\n T_est: {T_est}')
      print(f'p_true: {p_true},\n p_est: {p_est}')
  return T_est, p_est, T_true, p_true.reshape(-1,1)

def get_infl(args, state, val_data, unlabeled_data):
  """
  Get influence score of each unlabeled_data on val_data
  """
  logits, labels, groups = [], [], []
  num_samples = 0.0
  grad_avg = 0.0
  for example in val_data: # Need to run on the validation dataset to aviod the negative effect of distribution shift, e.g., DP is not robust to distribution shift.
    batch = preprocess_func_celeba(example, args)
    grads_each_sample = infl_step(state, batch)

    # moving average
    grad_avg = grad_avg * num_samples + jnp.sum(grads_each_sample, axis=0)
    num_samples += grads_each_sample.shape[0]
    grad_avg /= num_samples
  grad_avg = grad_avg.reshape(-1,1)
  for example in unlabeled_data:
    batch = preprocess_func_celeba(example, args)
    grads_each_sample = infl_step(state, batch)
    score = jnp.matmul(grads_each_sample) # bsz * 1
    # TODO





  return compute_metrics(
    logits=jnp.concatenate(logits),
    labels=jnp.concatenate(labels),
    groups=jnp.concatenate(groups),
  )


def test(args, state, data):
  """
  Test
  """
  logits, labels, groups = [], [], []
  for example in data:
    batch = preprocess_func_celeba(example, args)
    # batch = example
    logit= test_step(state, batch)
    logits.append(logit)
    labels.append(batch[args.label_key])
    groups.append(batch[args.group_key])

  return compute_metrics(
    logits=jnp.concatenate(logits),
    labels=jnp.concatenate(labels),
    groups=jnp.concatenate(groups),
  )
  # return None

def train(args):
  # setup
  set_global_seed()
  make_dirs(args)

  train_loader = load_celeba_dataset_torch(args, shuffle_files=False, split='train', batch_size=256)

  # ds_train = load_celeba_dataset(args, shuffle_files=False, batch_size=args.train_batch_size)
  # ds_test = load_celeba_dataset(args, shuffle_files=False, split='test', batch_size=args.test_batch_size)
  # indices = tf.data.Dataset.from_tensor_slices(tf.range(args.datasize))
  # total_dataset = tf.data.Dataset.zip((ds_train, indices)) # TODO: implement this with iterable dataset


  args.image_shape = args.img_size
  # setup
  model, model_linear = get_model(args)
  args.hidden_size = model_linear.hidden_size
  state = create_train_state(model, args)
  state_reg = create_train_state(model, args) # use the full model (ADMM)

  rec = init_recorder()

  
  
  # # /root/fair-eval/celeba/smile_gender_Facenet512_0.0_0.0.pt
  # if args.feature_extractor == 'None':
  #   T_rec = None
  #   data_path = f'/root/fair-eval/celeba/smile_gender_Facenet_0.0_0.0.pt'
  #   import torch
  #   data = torch.load(data_path)
  #   noisy_attribute = np.transpose(data['train_noisy_gender'])
  # else:
  #   data_path = f'/root/fair-eval/celeba/smile_gender_{args.feature_extractor}_0.0_0.0.pt'
  #   import torch
  #   data = torch.load(data_path)
  #   noisy_attribute = np.transpose(data['train_noisy_gender'])
  #   num_sample = noisy_attribute.shape[0]
  #   true_attribute = np.array(data['train_gender'])[:num_sample]
  #   y_pred = np.array(data['train_pred'])[:num_sample]
  #   y_true = np.array(data['train_label'])[:num_sample]
  #   args.max_iter = 1000
  #   args.G = 50
  #   T_est, p_est, T_true, p_true = get_T_p(args, noisy_attribute, lr = 0.1, true_attribute = true_attribute)
  #   # exit()
  #   data_eval = {'y_pred': y_pred,
  #             'y_true': y_true,
  #             'noisy_attribute': noisy_attribute,
  #             'true_attribute': true_attribute,
  #             'T_est': T_est,
  #             'p_est': p_est,
  #             'T_true': T_true,
  #             'p_true': p_true }
  #   T_est, p_est, T_true, p_true = [], [], [], []
  #   for k in np.unique(y_pred):
  #       loc = y_pred == k
  #       T_est_tmp, p_est_tmp, T_true_tmp, p_true_tmp = get_T_p(args, noisy_attribute[loc], lr = 0.1, true_attribute = true_attribute[loc])
  #       T_est.append(T_est_tmp)
  #       p_est.append(p_est_tmp)
  #       T_true.append(T_true_tmp)
  #       p_true.append(p_true_tmp)
  #   p = data_eval['p_est']
  #   T = T_est
  #   cnt = 0
  #   T_rec = []
  #   for k in np.unique(y_pred):  
  #       noisy_p = np.diag(np.array([np.mean((noisy_attribute==i)*1.0) for i in np.unique(noisy_attribute)]))
  #       diag_p_inv = np.linalg.inv(np.diag(p.reshape(-1)))
  #       T_trans_inv = np.linalg.inv(np.transpose(T[cnt]))

  #       correct_T = np.dot( np.dot(diag_p_inv, T_trans_inv),  noisy_p)
  #       T_rec.append(correct_T)
  #       cnt += 1


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
  for epoch_i in range(args.num_epochs):

    # get T during training 
    # T = None # TODO
    # T = T_rec
    # without conditional independence

        


    t = 0
    num_sample_cur = 0
    print(f'Epoch {epoch_i}')
    if epoch_i < args.warm_epoch: 
      print(f'warmup epoch = {epoch_i+1}/{args.warm_epoch}')
    if epoch_i == args.warm_epoch:
      state_reg = create_train_state(model, args, params=state.params) # use the full model
    for example in train_loader:
      bsz = example[args.feature_key].shape[0]
      # noisy_attribute_sel = noisy_attribute[num_sample_cur:num_sample_cur + bsz]
      num_sample_cur += bsz
      example = preprocess_func_celeba_torch(example, args, noisy_attribute = None)
      args = global_var.get_value('args')
      t += 1
      # load data
      if args.balance_batch:
        image, group, label = example[args.feature_key], example[args.group_key], example[args.label_key]
        num_a, num_b = jnp.sum((group == 0) * 1.0), jnp.sum((group == 1) * 1.0)
        min_num = min(num_a, num_b).astype(int)
        total_idx = jnp.arange(len(group))
        if min_num > 0:
          group_a = total_idx[group == 0]
          group_b = total_idx[group == 1]
          group_a = group_a.repeat(args.train_batch_size//2//len(group_a)+1)[:args.train_batch_size//2]
          group_b = group_b.repeat(args.train_batch_size//2//len(group_b)+1)[:args.train_batch_size//2]

          sel_idx = jnp.concatenate((group_a,group_b))
          batch = {args.feature_key: jnp.array(image[sel_idx]), args.label_key: jnp.array(label[sel_idx]), args.group_key: jnp.array(group[sel_idx])}
        else:
          print(f'current batch only contains one group')
          continue

      else:
        batch = example

      # train
      if args.method == 'plain':
        state, train_metric = train_step(state, batch)
      elif args.method in ['fix_lmd','dynamic_lmd']:
        # pdb.set_trace()
        state, train_metric, lmd = train_step(state, batch, lmd = lmd, T=None)
      elif args.method == 'admm':
        if epoch_i < args.warm_epoch: 
          train_step_warmup = get_train_step(method = 'plain')
          state, train_metric = train_step_warmup(state, batch)
          
        else:
          # print(lmd)
          state, state_reg, train_metric, lmd, loss, aux_all = train_step(state, state_reg, batch, lmd = lmd, mu = args.mu)
          # print(loss)
          loss = loss.item()
          loss_rec.append(loss)
          # print(jnp.sort(jnp.round(jnp.abs(aux_reg[0]-aux_reg[1]),2))[100:])
          # print(jnp.round(aux_reg[1],2))
          # print('-----')
          # print(f'lmd is {lmd}')
          # weight_params = jax.tree_util.tree_leaves(state.params)
          # weight_params_reg = jax.tree_util.tree_leaves(state_reg.params)
          # weight_para_vec = jnp.concatenate([x.reshape(-1) for x in weight_params]) 
          # weight_para_vec_reg = jnp.concatenate([x.reshape(-1) for x in weight_params_reg]) 
          # # print(f'[MAIN] model gap vector: {(weight_para_vec - weight_para_vec_reg)}' )
          # print(f'[MAIN] model gap max: {jnp.max(jnp.abs(weight_para_vec - weight_para_vec_reg))}' )
          # print('-----')

      rec = record_train_stats(rec, t-1, train_metric, 0)
     
      if t % args.log_steps == 0:
        # test
        
        test_metric = test(args, state, ds_test)
        rec, time_now = record_test(rec, t+args.datasize*epoch_i//args.train_batch_size, args.datasize*args.num_epochs//args.train_batch_size, time_now, time_start, train_metric, test_metric)
        # print(lmd)
        if args.method == 'admm':
          print(f'average train loss is {np.mean(loss_rec)}')
          loss_rec = [loss]
          if epoch_i >= args.warm_epoch: 
            aux_reg = aux_all[0]
            print(f'model confidence: {jnp.sort(jnp.round(jnp.abs(aux_reg[0][0]-0.5),2))[:50]}')
            print(f'ef: {jnp.abs(aux_reg[0][1])}, ez: {jnp.abs(aux_reg[0][2])}, efz: {jnp.abs(aux_reg[0][3])}')
            print(f'loss_conf: {(aux_reg[1])}, loss_reg: {(aux_reg[2])}, loss_model_gap: {(aux_reg[3])}')
            # weight_params = jax.tree_util.tree_leaves(state.params)
            # weight_params_reg = jax.tree_util.tree_leaves(state_reg.params)
            # weight_para_vec = jnp.concatenate([x.reshape(-1) for x in weight_params]) 
            # weight_para_vec_reg = jnp.concatenate([x.reshape(-1) for x in weight_params_reg]) 
            # print(f'model gap check: {jnp.sum((weight_para_vec - weight_para_vec_reg)**2)}' )

            aux_main = aux_all[1]
            print(f'[MAIN] model confidence: {jnp.sort(jnp.round(jnp.abs(aux_main[0][0]-0.5),2))[:50]}')
            print(f'[MAIN] ef: {jnp.abs(aux_main[0][1])}, ez: {jnp.abs(aux_main[0][2])}, efz: {jnp.abs(aux_main[0][3])}')
            print(f'[MAIN] loss_conf: {(aux_main[1])}, loss_reg: {(aux_main[2])}, loss_model_gap: {(aux_main[3])}')
            weight_params = jax.tree_util.tree_leaves(state.params)
            weight_params_reg = jax.tree_util.tree_leaves(state_reg.params)
            weight_para_vec = jnp.concatenate([x.reshape(-1) for x in weight_params]) 
            weight_para_vec_reg = jnp.concatenate([x.reshape(-1) for x in weight_params_reg]) 
            print(f'[MAIN] model gap check: {jnp.sum((weight_para_vec - weight_para_vec_reg)**2)}' )
        print(f'lmd is {lmd}')
            # pdb.set_trace()


    rec = save_checkpoint(args.save_dir, t+args.datasize*epoch_i//args.train_batch_size, state, rec, save=False)

  # wrap it up
  save_recorder(args.save_dir, rec)
  # return test_metric
