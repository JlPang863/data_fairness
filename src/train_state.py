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

def initialized(key, input_shape, model): 
  # if isinstance(input_shape, int):
  #   input_shape = (1, input_shape, input_shape, 3)
  @jax.jit
  def init(*args):
    return model.init(*args)
  variables = init({'params': key}, jnp.ones(input_shape))
  return variables['params'], variables['batch_stats']

def initialized_vit(key, input_shape, model):
  # if isinstance(input_shape, int):
  #   input_shape = (1, input_shape, input_shape, 3)

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

def create_train_state(model, args, params=None, return_opt = False):
  def custom_scheduler(init_lr):
    return optax.piecewise_constant_schedule(init_value=init_lr,
                                            boundaries_and_scales={1:1.0})
  default_lr = custom_scheduler(args.lr)
  rng = jax.random.PRNGKey(args.model_seed)
  lr_scheduler = None

  # instantiate optax optimizer
  try:
    opt_clsname = getattr(optax, args.opt['name'])
    opt_config = args.opt['config'].copy()

    # lr scheduler
    if args.scheduler is not None:
      scheduler_clsname = getattr(optax, args.scheduler['name'])
      lr_scheduler = scheduler_clsname(**args.scheduler['config'])
      opt_config['learning_rate'] = lr_scheduler
    else:
      opt_config['learning_rate'] = default_lr

    tx = opt_clsname(**opt_config)
  except:
    # default optimizer
    tx = optax.sgd(learning_rate=default_lr, momentum=args.momentum, nesterov=args.nesterov)

  if params is None:
    if 'vit' in args.model:
      params = initialized_vit(rng, args.input_shape, model)  
      batch_stats = None
    else:
      params, batch_stats = initialized(rng, args.input_shape, model)
  else:
    batch_stats = None

  state = TrainState.create(
      apply_fn=model.apply,
      params=params,
      tx=tx,
      batch_stats=batch_stats)
  if return_opt:
    return state, lr_scheduler
  else:
    return state



@jax.jit
def train_plain(state, batch): 
  """
  plain training
  """
  # args = global_var.get_value('args')
  loss_fn = get_loss_fn(state, batch, detaild_loss = False)
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  aux, grads = grad_fn(state.params)
  new_model_state, logits = aux[1]
  metrics = compute_metrics(logits=logits, labels=batch['label'], groups = batch['group'])
  if state.batch_stats:
    new_state = state.apply_gradients(grads=grads, batch_stats=new_model_state['batch_stats'])
  else:
    new_state = state.apply_gradients(grads=grads)
  return new_state, metrics

# @jax.jit
# def train_fix_lmd(state, batch, lmd): 
#   """
#   fixed-lambda training
#   """
#   args = global_var.get_value('args')
#   loss_fn = get_loss_lmd_fix(args, state, batch)
#   grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
#   aux, grads = grad_fn(state.params, lmd)
#   new_model_state, logits, _ = aux[1]
#   metrics = compute_metrics(logits=logits, labels=batch['label'], groups = batch['group'])
#   if state.batch_stats:
#     new_state = state.apply_gradients(grads=grads, batch_stats=new_model_state['batch_stats'])
#   else:
#     new_state = state.apply_gradients(grads=grads)
#   return new_state, metrics, lmd


@jax.jit
def train_dynamic_lmd_two_loader_warm(state, batch, batch_fair, lmd = 1.0, T = None, worst_group_id = 0): 
  """
  dynamic-lambda training
  """
  # pdb.set_trace()

  # args = global_var.get_value('args')
  loss_fn = get_loss_lmd_dynamic_two_loader_warm(state, batch, batch_fair, per_sample=False, T = T, worst_group_id = worst_group_id)
  
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

  aux, grads = grad_fn(state.params, lmd) # aux[0]: loss
  new_model_state, logits, logits_fair, lmd = aux[1]

  metrics_fair = compute_metrics_fair(logits=logits_fair, labels=batch_fair['label'], groups = batch_fair['group'])
  metrics = compute_metrics(logits=logits, labels=batch['label'], groups = None)
  if state.batch_stats:
    new_state = state.apply_gradients(grads=grads, batch_stats=new_model_state['batch_stats'])
  else:
    new_state = state.apply_gradients(grads=grads)
  # print('grad backward done')
  return new_state, metrics, metrics_fair, lmd


@jax.jit
def train_dynamic_lmd_two_loader(state, batch, batch_fair, lmd = 1.0, T = None, worst_group_id = 0): 
  """
  dynamic-lambda training
  """
  # pdb.set_trace()

  # args = global_var.get_value('args')
  loss_fn = get_loss_lmd_dynamic_two_loader(state, batch, batch_fair, per_sample=False, T = T, worst_group_id = worst_group_id)
  
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

  aux, grads = grad_fn(state.params, lmd) # aux[0]: loss
  new_model_state, logits, logits_fair, lmd = aux[1]

  metrics_fair = compute_metrics_fair(logits=logits_fair, labels=batch_fair['label'], groups = batch_fair['group'])
  metrics = compute_metrics(logits=logits, labels=batch['label'], groups = None)
  if state.batch_stats:
    new_state = state.apply_gradients(grads=grads, batch_stats=new_model_state['batch_stats'])
  else:
    new_state = state.apply_gradients(grads=grads)
  # print('grad backward done')
  return new_state, metrics, metrics_fair, lmd

# @jax.jit
# def train_dynamic_lmd(state, batch, lmd = 1.0, T = None): 
#   """
#   dynamic-lambda training
#   """
#   # pdb.set_trace()

#   args = global_var.get_value('args')
#   loss_fn = get_loss_lmd_dynamic(state, batch, per_sample=False, T = T)
  
#   grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

#   aux, grads = grad_fn(state.params, lmd) # aux[0]: loss
#   new_model_state, logits, lmd = aux[1]

#   metrics = compute_metrics(logits=logits, labels=batch['label'], groups = batch['group'])
#   if state.batch_stats:
#     new_state = state.apply_gradients(grads=grads, batch_stats=new_model_state['batch_stats'])
#   else:
#     new_state = state.apply_gradients(grads=grads)
#   print('grad backward done')
#   return new_state, metrics, lmd


def get_train_step(method):
  """
  Train for a single step.
  Method: plain, fix_lmd, dynamic_lmd, admm 
  """
  if method == 'plain':
    return train_plain
  elif method == 'fix_lmd':
    raise NameError('Undefined')
    # return train_fix_lmd
  elif method == 'dynamic_lmd':
    return train_dynamic_lmd_two_loader, train_dynamic_lmd_two_loader_warm
    # return train_dynamic_lmd


@jax.jit
def test_step(state, batch):
  """
  Test for a single step.
  """
  # args = global_var.get_value('args')
  if state.batch_stats:
    variables = {'params': state.params, 'batch_stats': state.batch_stats}
  else:
    variables = {'params': state.params}
  logits = state.apply_fn(
      variables, batch['feature'], train=False, mutable=False)
  if len(logits) == 2:
    logits = logits[0]
  return logits


@jax.jit
def infl_step(state, batch):
  """
  Get grads for infl scores
  Return:
    Grads: (model_size,)
  """
  
  


  # loss_fn_per_sample = get_loss_lmd_dynamic(state, batch, per_sample=True)
  loss_fn = get_loss_fn(state, batch, per_sample=False)

  
  grads_tree, aux = jax.jacrev(loss_fn, argnums=0, has_aux=True)(state.params)
  # grads_tree, aux = our_jacrev(loss_fn, argnums=0, has_aux=True)(state.params)
  
  import pdb
  pdb.set_trace()
  # grad_flat_tree = jax.tree_util.tree_leaves(grads_tree)
  grad_flat_tree = jax.tree_util.tree_leaves(grads_tree[0])
  grad_org_flat_tree = jax.tree_util.tree_leaves(grads_tree[1])

  args = global_var.get_value('args')
  if args.sel_layers > 0:
    sel_layers = grad_flat_tree[:args.sel_layers]
    sel_layers_org = grad_org_flat_tree[:args.sel_layers]
  else:
    sel_layers = grad_flat_tree[args.sel_layers:]
    sel_layers_org = grad_org_flat_tree[args.sel_layers:]
  print(f'selected layers: {sel_layers}')
  grads = jnp.concatenate([x.reshape(-1) for x in sel_layers], axis=-1)
  grads_org = jnp.concatenate([x.reshape(-1) for x in sel_layers_org], axis=-1)


  return grads, grads_org


@jax.jit
def infl_step_per_sample(state, batch):
  """
  Get grads for infl scores of each sample.
  Return:
    Grads: (bsz, model_size)
  """

  # loss_fn_per_sample = get_loss_lmd_dynamic(state, batch, per_sample=True)
  loss_fn_per_sample = get_loss_fn(state, batch, per_sample=True)

  
  grads_per_sample_tree, aux = jax.jacrev(loss_fn_per_sample, argnums=0, has_aux=True)(state.params)

  grad_flat_tree = jax.tree_util.tree_leaves(grads_per_sample_tree)

  # if batch['label'] is None:
  #   grads_per_sample = jnp.concatenate([x.reshape(grad_flat_tree[-1].shape[0], grad_flat_tree[-1].shape[1], -1) for x in grad_flat_tree[-4:]], axis=-1)  # last two layers
  # else:
  #   grads_per_sample = jnp.concatenate([x.reshape(batch['feature'].shape[0],-1) for x in grad_flat_tree[-4:]], axis=-1)
    


  args = global_var.get_value('args')
  if args.sel_layers > 0:
    sel_layers = grad_flat_tree[:args.sel_layers]
  else:
    sel_layers = grad_flat_tree[args.sel_layers:]

  if batch['label'] is None:
    grads_per_sample = jnp.concatenate([x.reshape(sel_layers[-1].shape[0], sel_layers[-1].shape[1], -1) for x in sel_layers], axis=-1) # last layer
  else:
    grads_per_sample = jnp.concatenate([x.reshape(batch['feature'].shape[0],-1) for x in sel_layers], axis=-1)

  return grads_per_sample, aux[1] # grad and logits

@jax.jit
def infl_step_fair(state, batch):
  """
  Get grads for infl scores of each sample.
  Return:
    Grads: (model_size,)
  """


  loss_fn_per_sample = get_loss_fair(state, batch)

  
  grads_per_sample_tree, aux = jax.jacrev(loss_fn_per_sample, argnums=0, has_aux=True)(state.params)
  
  grad_flat_tree = jax.tree_util.tree_leaves(grads_per_sample_tree)

  # grads_per_sample = jnp.concatenate([x.reshape(-1) for x in grad_flat_tree[-4:]], axis=-1)
  args = global_var.get_value('args')
  if args.sel_layers > 0:
    sel_layers = grad_flat_tree[:args.sel_layers]
  else:
    sel_layers = grad_flat_tree[args.sel_layers:]

  grads_per_sample = jnp.concatenate([x.reshape(-1) for x in sel_layers], axis=-1)

  # import pdb
  # pdb.set_trace()
    

  return grads_per_sample