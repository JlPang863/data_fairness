import optax
from .metrics import *
from . import global_var




def get_loss_fn(args, state, batch):
  constraints_confidence = constraints_dict[args.conf]
  def loss_fn(params):
    if state.batch_stats:
      logits, new_model_state = state.apply_fn({'params': params, 'batch_stats': state.batch_stats}, batch[args.feature_key], mutable=['batch_stats'])
    else:
      logits, new_model_state = state.apply_fn({'params': params}, batch[args.feature_key], mutable=['batch_stats'])
    if len(logits) == 2: # logits and embeddings
      logits = logits[0]
    loss = cross_entropy_loss(logits=logits, labels=batch[args.label_key])
    loss += constraints_confidence(logits)
    return loss, (new_model_state, logits)
  return loss_fn

def get_loss_lmd_fix(args, state, batch):
  constraints_fair = constraints_dict[args.metric]
  constraints_confidence = constraints_dict[args.conf]
  def loss_fn(params, lmd):
    if state.batch_stats:
      logits, new_model_state = state.apply_fn({'params': params, 'batch_stats': state.batch_stats}, batch[args.feature_key], mutable=['batch_stats'])
    else:
      logits, new_model_state = state.apply_fn({'params': params}, batch[args.feature_key], mutable=['batch_stats'])
    if len(logits) == 2: # logits and embeddings
      logits = logits[0]
    loss_reg, _ = constraints_fair(logits, batch[args.group_key], batch[args.label_key])
    loss = cross_entropy_loss(logits=logits, labels=batch[args.label_key]) + lmd * jnp.sum(jnp.abs(loss_reg))
    loss += constraints_confidence(logits)
    return loss, (new_model_state, logits, lmd)
  return loss_fn

def get_loss_lmd_dynamic(state, batch, per_sample = False, T = None):
  args = global_var.get_value('args')
  mu = args.mu
  constraints_fair = constraints_dict[args.metric]
  constraints_confidence = constraints_dict[args.conf]
  def loss_fn(params, lmd): 
    if state.batch_stats:
        logits, new_model_state = state.apply_fn({'params': params, 'batch_stats': state.batch_stats}, batch[args.feature_key], mutable=['batch_stats'])
    else:
        logits, new_model_state = state.apply_fn({'params': params}, batch[args.feature_key], mutable=['batch_stats'])
    if len(logits) == 2: # logits and embeddings
        logits = logits[0]
    loss_reg, _ = constraints_fair(logits, batch[args.group_key], batch[args.label_key], T = T)
    lmd = lmd + mu * loss_reg 
    loss = cross_entropy_loss(logits=logits, labels=batch[args.label_key]) + jnp.sum(mu/2 * loss_reg**2) + jnp.sum(lmd * loss_reg)
    loss += constraints_confidence(logits)
    return loss, (new_model_state, logits, lmd)

  def loss_fn_per_sample(params, lmd): 
    if state.batch_stats:
        logits, new_model_state = state.apply_fn({'params': params, 'batch_stats': state.batch_stats}, batch[args.feature_key], mutable=['batch_stats'])
    else:
        logits, new_model_state = state.apply_fn({'params': params}, batch[args.feature_key], mutable=['batch_stats'])
    if len(logits) == 2: # logits and embeddings
        logits = logits[0]
    loss_reg, _ = constraints_fair(logits, batch[args.group_key], batch[args.label_key]) # TODO: per sample
    lmd = lmd + mu * loss_reg 
    loss = cross_entropy_loss_per_sample(logits=logits, labels=batch[args.label_key]) + jnp.sum(mu/2 * loss_reg**2) + jnp.sum(lmd * loss_reg)
    loss += constraints_confidence(logits)
    return loss, (new_model_state, logits, lmd)
  if per_sample:
    return loss_fn_per_sample
  else:
    return loss_fn


def get_loss_admm_org(state, batch):
  args = global_var.get_value('args')
  mu = args.mu
  def loss_fn(params, params_reg, lmd):
    if state.batch_stats:
      logits, new_model_state = state.apply_fn({'params': params, 'batch_stats': state.batch_stats}, batch[args.feature_key], mutable=['batch_stats'])
    else:
      logits, new_model_state = state.apply_fn({'params': params}, batch[args.feature_key], mutable=['batch_stats'])
    if len(logits) == 2: # logits and embeddings
      logits, _ = logits
    else:
      raise NameError('Must have embeddings')

    # get weights
    weight_params = jax.tree_util.tree_leaves(params)
    weight_params_reg = jax.tree_util.tree_leaves(params_reg)
    weight_para_vec = (jnp.concatenate([x.reshape(-1) for x in weight_params]))
    weight_para_vec_reg = (jnp.concatenate([x.reshape(-1) for x in weight_params_reg]))

    loss_ce = cross_entropy_loss(logits=logits, labels=batch[args.label_key])
    loss_lmd = jnp.sum(lmd * (weight_para_vec - weight_para_vec_reg))
    loss_reg = mu/2 * jnp.sum(jnp.abs(weight_para_vec - weight_para_vec_reg)**2)
    loss = loss_ce + loss_lmd + loss_reg

    return loss, (new_model_state, logits)
  return loss_fn

def get_loss_admm_reg(state, state_reg, batch):
  mu = args.mu
  args = global_var.get_value('args')
  constraints_fair = constraints_dict[args.metric]
  constraints_confidence = constraints_dict[args.conf]
  def loss_fn(params, params_reg, lmd):
    if state.batch_stats:
      logits_main = state.apply_fn({'params': params, 'batch_stats': state.batch_stats}, batch[args.feature_key], mutable=False, train=False)
    else:
      logits_main = state.apply_fn({'params': params}, batch[args.feature_key], mutable=False, train=False)
    if len(logits_main) == 2: # logits and embeddings
      logits_main, _ = logits_main
    loss_main, aux_main = constraints_fair(logits_main, batch[args.group_key], batch[args.label_key])
    loss_conf_main = constraints_confidence_entropy(logits_main, batch[args.group_key])
    # if len(logits) == 2: # logits and embeddings
    #   _, embeddings = logits
    # else:
    #   raise NameError('Must have embeddings')

    # if state_reg.batch_stats:
    #   logits, new_model_state_reg = state_reg.apply_fn({'params': params_reg, 'batch_stats': state_reg.batch_stats}, embeddings, mutable=['batch_stats'])
    # else:
    #   logits, new_model_state_reg = state_reg.apply_fn({'params': params_reg}, embeddings, mutable=['batch_stats'])

    if state_reg.batch_stats:
      logits, new_model_state_reg = state_reg.apply_fn({'params': params_reg, 'batch_stats': state_reg.batch_stats}, batch[args.feature_key], mutable=['batch_stats'])
    else:
      logits, new_model_state_reg = state_reg.apply_fn({'params': params_reg}, batch[args.feature_key], mutable=['batch_stats'])
    if len(logits) == 2: # logits and embeddings
      logits = logits[0]

    # get weights
    # weight_params = jax.tree_util.tree_leaves(params['head']) # only use the last linear layer
    weight_params = jax.tree_util.tree_leaves(params)
    weight_params_reg = jax.tree_util.tree_leaves(params_reg)
    weight_para_vec = (jnp.concatenate([x.reshape(-1) for x in weight_params]) )
    weight_para_vec_reg = (jnp.concatenate([x.reshape(-1) for x in weight_params_reg]) )


    loss_constraint, aux_reg = constraints_fair(logits, batch[args.group_key], batch[args.label_key])
    # loss_reg, aux_reg = constraints_dp_ranking(logits, batch[args.group_key])
    # loss_reg, aux_reg = constraints_dp(logits, batch[args.group_key])
    loss_conf = constraints_confidence_entropy(logits, batch[args.group_key])
    # loss_two_net_gap = mu/2 * jnp.sum((weight_para_vec - weight_para_vec_reg + 1/mu * lmd)**2)

    loss_lmd = jnp.sum(lmd * (weight_para_vec - weight_para_vec_reg))
    loss_reg = mu/2 * jnp.sum(jnp.abs(weight_para_vec - weight_para_vec_reg)**2)
    # loss = jnp.abs(constraints_dp(logits, batch[args.group_key])) + constraints_confidence(logits, batch[args.group_key]) + mu/2 * jnp.sum((weight_para_vec - weight_para_vec_reg + 1/mu * lmd)**2)
    loss =  jnp.abs(loss_constraint) + loss_lmd + loss_reg

    loss += constraints_confidence(logits)
    aux_reg_out = (aux_reg,loss_conf,jnp.abs(loss_constraint),loss_reg)
    aux_main_out = (aux_main,loss_conf_main,jnp.abs(loss_main),loss_reg)
    return loss, (new_model_state_reg, logits, lmd, (aux_reg_out,aux_main_out))
  return loss_fn