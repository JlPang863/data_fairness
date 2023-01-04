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
  # args = global_var.get_value('args')
  loss_fn = get_loss_fn(state, batch)
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
  metrics = compute_metrics(logits=logits, labels=batch['label'], groups = batch['group'])
  if state.batch_stats:
    new_state = state.apply_gradients(grads=grads, batch_stats=new_model_state['batch_stats'])
  else:
    new_state = state.apply_gradients(grads=grads)
  print('grad backward done')
  return new_state, metrics, lmd


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
    return train_dynamic_lmd


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
  
  grad_flat_tree = jax.tree_util.tree_leaves(grads_tree)


  grads = jnp.concatenate([x.reshape(-1) for x in grad_flat_tree[-4:]], axis=-1)


  return grads


@jax.jit
def infl_step_per_sample(state, batch):
  """
  Get grads for infl scores of each sample.
  Return:
    Grads: (bsz, model_size)
  """

  # loss_fn_per_sample = get_loss_lmd_dynamic(state, batch, per_sample=True)
  loss_fn_per_sample = get_loss_fn(state, batch, per_sample=True)

  
  grads_per_sample_tree, grads_org_per_sample_tree, aux = jax.jacrev(loss_fn_per_sample, argnums=(0,1), has_aux=True)(state.params)
  pdb.set_trace()
  # grads_per_sample_tree, aux = our_jacrev(loss_fn_per_sample, argnums=0, has_aux=True)(state.params)
  
  grad_flat_tree = jax.tree_util.tree_leaves(grads_per_sample_tree)
  grads_org_per_sample_tree = jax.tree_util.tree_leaves(grads_org_per_sample_tree)


  if batch['label'] is None:
    grads_per_sample = jnp.concatenate([x.reshape(grad_flat_tree[-1].shape[0], grad_flat_tree[-1].shape[1], -1) for x in grad_flat_tree[-4:]], axis=-1)
    grads_org_per_sample = jnp.concatenate([x.reshape(grads_org_per_sample_tree[-1].shape[0], grads_org_per_sample_tree[-1].shape[1], -1) for x in grads_org_per_sample_tree[-4:]], axis=-1)
  else:
    grads_per_sample = jnp.concatenate([x.reshape(batch['feature'].shape[0],-1) for x in grad_flat_tree[-4:]], axis=-1)
    grads_org_per_sample = jnp.concatenate([x.reshape(batch['feature'].shape[0],-1) for x in grads_org_per_sample_tree[-4:]], axis=-1)


  return grads_per_sample, grads_org_per_sample, aux[1] # grad and logits

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

  grads_per_sample = jnp.concatenate([x.reshape(-1) for x in grad_flat_tree[-4:]], axis=-1)
    

  return grads_per_sample