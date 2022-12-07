from ast import arg
from typing import Any
import jax
import optax
from flax.training import train_state 
from jax import numpy as jnp
from .loss_func import *
from .metrics import compute_metrics
from . import global_var
from functools import partial


def our_jacrev(f):
    def jacfun(x):
        y, vjp_fun = jax.vjp(f, x)
        # Use vmap to do a matrix-Jacobian product.
        # Here, the matrix is the Euclidean basis, so we get all
        # entries in the Jacobian at once. 
        J, = jax.vmap(vjp_fun, in_axes=0)(jnp.eye(len(y)))
        return J
    return jacfun

class TrainState(train_state.TrainState):
  batch_stats: Any

########################################################################################################################
#  Train
########################################################################################################################

def initialized(key, image_size, model): # TODO: image_size --> input_shape
  input_shape = (1, image_size, image_size, 3)
  @jax.jit
  def init(*args):
    return model.init(*args)
  variables = init({'params': key}, jnp.ones(input_shape))
  return variables['params'], variables['batch_stats']

def initialized_vit(key, image_size, model):
  input_shape = (1, image_size, image_size, 3)
  @jax.jit
  def init(*args):
    return model.init(*args)
  variables = init({'params': key}, jnp.ones(input_shape))
  return variables['params']

def create_train_state_linear(model, args, params=None): # TODO: will be removed
  rng = jax.random.PRNGKey(args.model_seed)

  # tx = optax.sgd(args.lr, args.momentum)
  tx = optax.sgd(args.lr, args.momentum, nesterov=True)
  if params is None:
    params = model.init(rng, jnp.ones(args.hidden_size))['params']

  state = TrainState.create(
      apply_fn=model.apply,
      params=params,
      tx=tx,
      batch_stats=None)
  
  return state

def create_train_state(model, args, params=None):
  rng = jax.random.PRNGKey(args.model_seed)

  # tx = optax.sgd(args.lr, args.momentum)
  # tx = optax.sgd(args.lr, args.momentum, nesterov=True)
  # instantiate optax optimizer
  try:
    opt_clsname = getattr(optax, args.opt['name'])
    opt_config = args.opt['config'].copy()

    # lr scheduler
    if args.scheduler is not None:
      scheduler_clsname = getattr(optax, args.scheduler['name'])
      lr_scheduler = scheduler_clsname(**args.scheduler['config'])
      opt_config['learning_rate'] = lr_scheduler

    tx = opt_clsname(**opt_config)
  except:
    # default optimizer
    tx = optax.sgd(learning_rate=args.lr, momentum=args.momentum, nesterov=args.nesterov)

  if params is None:
    if 'vit' in args.model:
      params = initialized_vit(rng, args.image_shape, model)  
      batch_stats = None
    else:
      params, batch_stats = initialized(rng, args.image_shape, model)
  else:
    batch_stats = None

  state = TrainState.create(
      apply_fn=model.apply,
      params=params,
      tx=tx,
      batch_stats=batch_stats)
  
  return state



@jax.jit
def train_plain(state, batch): 
  """
  plain training
  """
  args = global_var.get_value('args')
  loss_fn = get_loss_fn(args, state, batch)
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  aux, grads = grad_fn(state.params)
  new_model_state, logits = aux[1]
  metrics = compute_metrics(logits=logits, labels=batch[args.label_key], groups = batch[args.group_key])
  if state.batch_stats:
    new_state = state.apply_gradients(grads=grads, batch_stats=new_model_state['batch_stats'])
  else:
    new_state = state.apply_gradients(grads=grads)
  return new_state, metrics

@jax.jit
def train_fix_lmd(state, batch, lmd): 
  """
  fixed-lambda training
  """
  args = global_var.get_value('args')
  loss_fn = get_loss_lmd_fix(args, state, batch)
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  aux, grads = grad_fn(state.params, lmd)
  new_model_state, logits, _ = aux[1]
  metrics = compute_metrics(logits=logits, labels=batch[args.label_key], groups = batch[args.group_key])
  if state.batch_stats:
    new_state = state.apply_gradients(grads=grads, batch_stats=new_model_state['batch_stats'])
  else:
    new_state = state.apply_gradients(grads=grads)
  return new_state, metrics, lmd

# @partial(jax.jit, static_argnames=['args'])
@jax.jit
def train_dynamic_lmd(state, batch, lmd = 1.0, T = None): 
  """
  dynamic-lambda training
  """
  # pdb.set_trace()

  args = global_var.get_value('args')
  loss_fn = get_loss_lmd_dynamic(state, batch, per_sample=False, T = T)
  
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

  aux, grads = grad_fn(state.params, lmd) # aux[0]: loss
  new_model_state, logits, lmd = aux[1]

  # loss_fn_per_sample = get_loss_lmd_dynamic(state, batch, per_sample=True)
  # grads_per_sample_tree, aux = jax.jacrev(loss_fn_per_sample, argnums=0, has_aux=True)(state.params, lmd)
  # grad_flat_tree = jax.tree_util.tree_leaves(grads_per_sample_tree)
  # grads_per_sample = jnp.concatenate([x.reshape(x.shape[0],-1) for x in grad_flat_tree], axis=-1) 
  # print(grads_per_sample.shape)



  # if per_sample:
    # loss_fn = get_loss_lmd_dynamic(state, batch, per_sample=False)
    # grads_per_sample_tree, aux = jax.jacrev(loss_fn, argnums=0, has_aux=True)(state.params, lmd)
    # grads_per_sample_tree, aux = our_jacrev(loss_fn, argnums=0, has_aux=True)(state.params, lmd)
    
    # new_model_state, logits, lmd = aux
    # grad_flat_tree = jax.tree_util.tree_leaves(grads_per_sample_tree)
    # grads_per_sample = jnp.concatenate([x.reshape(-1) for x in grad_flat_tree]) 
    # print(grads_per_sample.shape)
    # loss_fn = get_loss_lmd_dynamic(state, batch, per_sample=False)
    # grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    
    # aux, grads = grad_fn(state.params, lmd) # aux[0]: loss
    # new_model_state, logits, lmd = aux[1]
    # grads = jax.tree_util.tree_map(lambda x: jnp.mean(x,0), grads_per_sample_tree)
    # 
    
    # grads = grads_per_sample_tree
    # del grads_per_sample_tree
  # else:
    
  # pdb.set_trace()
  metrics = compute_metrics(logits=logits, labels=batch[args.label_key], groups = batch[args.group_key])
  if state.batch_stats:
    new_state = state.apply_gradients(grads=grads, batch_stats=new_model_state['batch_stats'])
  else:
    new_state = state.apply_gradients(grads=grads)
  print('grad backward down')
  return new_state, metrics, lmd

@jax.jit
def train_dynamic_admm(state, state_reg, batch, lmd = 1): 
  """
  ADMM dual network training
  state: full model
  state_reg: last linear layer
  """
  args = global_var.get_value('args')
  # ADMM Step 1:
  loss_fn = get_loss_admm_org(state, batch)
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True, argnums = 0) # differentiate wrt position 0
  aux, grads = grad_fn(state.params, state_reg.params, lmd=lmd)
  new_model_state, logits= aux[1]
  metrics = compute_metrics(logits=logits, labels=batch[args.label_key], groups = batch[args.group_key])
  if state.batch_stats:
    new_state = state.apply_gradients(grads=grads, batch_stats=new_model_state['batch_stats'])
  else:
    new_state = state.apply_gradients(grads=grads)
  
  # ADMM Step 2:
  loss_reg = get_loss_admm_reg(new_state, state_reg, batch)
  
  grad_reg = jax.value_and_grad(loss_reg, has_aux=True, argnums=1) # differentiate wrt position 1
  aux, grads = grad_reg(new_state.params, state_reg.params, lmd=lmd) # grad is calculated wrt the first arg 
  new_model_state_reg, _, _, aux_reg = aux[1]
  # print(aux[0])
  # metrics = compute_metrics(logits=logits, labels=batch[args.label_key], groups = batch[args.group_key])
  if state.batch_stats:
    new_state_reg = state_reg.apply_gradients(grads=grads, batch_stats=new_model_state_reg['batch_stats'])
  else:
    new_state_reg = state_reg.apply_gradients(grads=grads)

  # ADMM Step 3 update lmd  # very unstable
  # weight_params = jax.tree_util.tree_leaves(new_state.params)
  # weight_params_reg = jax.tree_util.tree_leaves(new_state_reg.params)
  # weight_para_vec = (jnp.concatenate([x.reshape(-1) for x in weight_params]) )
  # weight_para_vec_reg = (jnp.concatenate([x.reshape(-1) for x in weight_params_reg]) )
  # lmd = lmd + mu * (weight_para_vec - weight_para_vec_reg)

  return new_state, new_state_reg, metrics, lmd, aux[0], aux_reg


def get_train_step(method):
  """
  Train for a single step.
  Method: plain, fix_lmd, dynamic_lmd, admm 
  """
  if method == 'plain':
    return train_plain
  elif method == 'fix_lmd':
    return train_fix_lmd
  elif method == 'dynamic_lmd':
    return train_dynamic_lmd
  elif method == 'admm':
    return train_dynamic_admm

@jax.jit
def test_step(state, batch):
  """
  Test for a single step.
  """
  args = global_var.get_value('args')
  if state.batch_stats:
    variables = {'params': state.params, 'batch_stats': state.batch_stats}
  else:
    variables = {'params': state.params}
  logits = state.apply_fn(
      variables, batch[args.feature_key], train=False, mutable=False)
  if len(logits) == 2:
    logits = logits[0]
  return logits


@jax.jit
def infl_step(state, batch):
  """
  Get grads for infl scores of each sample.
  Return:
    Grads: bsz * model_size
  """
  args = global_var.get_value('args')
  if state.batch_stats:
    variables = {'params': state.params, 'batch_stats': state.batch_stats}
  else:
    variables = {'params': state.params}

  loss_fn_per_sample = get_loss_lmd_dynamic(state, batch, per_sample=True)
  grads_per_sample_tree, aux = jax.jacrev(loss_fn_per_sample, argnums=0, has_aux=True)(state.params, lmd=0)
  grad_flat_tree = jax.tree_util.tree_leaves(grads_per_sample_tree)
  grads_per_sample = jnp.concatenate([x.reshape(x.shape[0],-1) for x in grad_flat_tree], axis=-1) 

  return grads_per_sample