import optax
from .metrics import *
from . import global_var



def get_loss_fair(state, batch, T = None):
  args = global_var.get_value('args')
  mu = args.mu
  constraints_fair = constraints_dict[args.metric]
  constraints_confidence = constraints_dict[args.conf]
  def loss_fn(params, lmd = 0.0): 
    if state.batch_stats:
        logits, new_model_state = state.apply_fn({'params': params, 'batch_stats': state.batch_stats}, batch['feature'], mutable=['batch_stats'])
    else:
        logits, new_model_state = state.apply_fn({'params': params}, batch['feature'], mutable=['batch_stats'])
    if len(logits) == 2: # logits and embeddings
        logits = logits[0]
    if T is None:
      loss_reg, _ = constraints_fair(logits, batch['group'], batch['label'])
    else:
      loss_reg, _ = constraints_fair(logits, batch['group'], batch['label'], T = T)
    # lmd = lmd + mu * loss_reg # TODO
    loss = jnp.sum(mu/2 * loss_reg**2) # + jnp.sum(lmd * loss_reg)
    if not args.train_conf:
      loss += constraints_confidence(logits)
    return loss, (new_model_state, logits, lmd)

  return loss_fn


def get_loss_fn(state, batch, per_sample = False, detaild_loss = True):

  args = global_var.get_value('args')
  constraints_confidence = constraints_dict[args.conf]
  constraints_confidence_per_sample = constraints_dict_per_sample[args.conf]
  def loss_fn(params):
    if state.batch_stats:
      logits, new_model_state = state.apply_fn({'params': params, 'batch_stats': state.batch_stats}, batch['feature'], mutable=['batch_stats'])
    else:
      logits, new_model_state = state.apply_fn({'params': params}, batch['feature'], mutable=['batch_stats'])
    if len(logits) == 2: # logits and embeddings
      logits = logits[0]
    
    loss = cross_entropy_loss(logits=logits, labels=batch['label'])
    # print(f'close to the weights: args.curr_epoch: {args.curr_epoch}; warm_epoch: {args.warm_epoch:}')
    ###############################
    if args.strategy == 7:# and args.curr_epoch >= args.warm_epoch:
      #loss = loss * args.weights
      weights = 20
      # print(f"Baseline JTT weight: {weights}!!")
      loss = loss * weights
    ###############################
    # import pdb
    # pdb.set_trace()
    loss_org = loss
    if args.train_conf:
      loss += constraints_confidence(logits)
    if detaild_loss:
      return (loss, loss_org), (new_model_state, logits)
    else:
      return loss, (new_model_state, logits)

  def loss_fn_per_sample(params): 
    if state.batch_stats:
        logits, new_model_state = state.apply_fn({'params': params, 'batch_stats': state.batch_stats}, batch['feature'], mutable=['batch_stats'])
    else:
        logits, new_model_state = state.apply_fn({'params': params}, batch['feature'], mutable=['batch_stats'])
    if len(logits) == 2: # logits and embeddings
        logits = logits[0]
    loss = cross_entropy_loss_per_sample(logits=logits, labels=batch['label']) 
    # loss_org = loss
    if args.train_conf:
      loss += constraints_confidence_per_sample(logits)
    return loss, (new_model_state, logits)

  if per_sample:
    return loss_fn_per_sample
  else:
    return loss_fn




def get_loss_lmd_fix(args, state, batch):
  constraints_fair = constraints_dict[args.metric]
  constraints_confidence = constraints_dict[args.conf]
  def loss_fn(params, lmd):
    if state.batch_stats:
      logits, new_model_state = state.apply_fn({'params': params, 'batch_stats': state.batch_stats}, batch['feature'], mutable=['batch_stats'])
    else:
      logits, new_model_state = state.apply_fn({'params': params}, batch['feature'], mutable=['batch_stats'])
    if len(logits) == 2: # logits and embeddings
      logits = logits[0]
    loss_reg, _ = constraints_fair(logits, batch['group'], batch['label'])
    loss = cross_entropy_loss(logits=logits, labels=batch['label']) + lmd * jnp.sum(jnp.abs(loss_reg))
    loss += constraints_confidence(logits)
    return loss, (new_model_state, logits, lmd)
  return loss_fn

def get_loss_lmd_dynamic_two_loader_warm(state, batch, batch_fair, per_sample = False, T = None, worst_group_id = 0):
  args = global_var.get_value('args')
  mu = args.mu
  constraints_fair = constraints_dict[args.metric]

  def loss_fn(params, lmd): 
    if state.batch_stats:
        logits, new_model_state = state.apply_fn({'params': params, 'batch_stats': state.batch_stats}, batch['feature'], mutable=['batch_stats'])
        logits_fair, _ = state.apply_fn({'params': params, 'batch_stats': state.batch_stats}, batch_fair['feature'], mutable=['batch_stats'])
    else:
        logits, new_model_state = state.apply_fn({'params': params}, batch['feature'], mutable=['batch_stats'])
        logits_fair, _ = state.apply_fn({'params': params}, batch_fair['feature'], mutable=['batch_stats'])
    if len(logits) == 2: # logits and embeddings
        logits = logits[0]
        logits_fair = logits_fair[0]

    loss_reg, _ = constraints_fair(logits_fair, batch_fair['group'], batch_fair['label'], T = T)
    lmd = lmd + mu * loss_reg 
    loss_fair = jnp.sum(mu/2 * loss_reg**2) + jnp.sum(lmd * loss_reg)
    if args.exp == 1:
      loss_fair += cross_entropy_loss(logits=logits_fair, labels=batch_fair['label'])
    elif args.exp == 2:
      weight = (jnp.repeat((batch_fair['label'] == worst_group_id).reshape(-1,1), 2, axis=1) * 1.0) + 1e-5
      loss_fair += cross_entropy_loss(logits=logits * weight, labels=batch_fair['label'])
    elif args.exp == 3:
      pass

    loss = cross_entropy_loss(logits=logits, labels=batch['label']) + loss_fair
    
    return loss, (new_model_state, logits, logits_fair, lmd)

  return loss_fn

def get_loss_lmd_dynamic_two_loader(state, batch, batch_fair, per_sample = False, T = None, worst_group_id = 0):
  args = global_var.get_value('args')
  mu = args.mu
  constraints_fair = constraints_dict[args.metric]
  constraints_confidence = constraints_dict[args.conf]

  def loss_fn(params, lmd): 
    if state.batch_stats:
        logits, new_model_state = state.apply_fn({'params': params, 'batch_stats': state.batch_stats}, batch['feature'], mutable=['batch_stats'])
        logits_fair, _ = state.apply_fn({'params': params, 'batch_stats': state.batch_stats}, batch_fair['feature'], mutable=['batch_stats'])
    else:
        logits, new_model_state = state.apply_fn({'params': params}, batch['feature'], mutable=['batch_stats'])
        logits_fair, _ = state.apply_fn({'params': params}, batch_fair['feature'], mutable=['batch_stats'])
    if len(logits) == 2: # logits and embeddings
        logits = logits[0]
        logits_fair = logits_fair[0]
        
    if T is None:
      loss_reg, _ = constraints_fair(logits_fair, batch_fair['group'], batch_fair['label'])

    else:
      loss_reg, _ = constraints_fair(logits_fair, batch_fair['group'], batch_fair['label'], T = T)
    lmd = lmd + mu * loss_reg 
    loss_fair = jnp.sum(mu/2 * loss_reg**2) + jnp.sum(lmd * loss_reg)
    if args.exp == 1:
      loss_fair += cross_entropy_loss(logits=logits_fair, labels=batch_fair['label'])
    elif args.exp == 2:
      weight = (jnp.repeat((batch_fair['label'] == worst_group_id).reshape(-1,1), 2, axis=1) * 1.0) + 1e-5
      loss_fair += cross_entropy_loss(logits=logits * weight, labels=batch_fair['label'])
    elif args.exp == 3:
      pass

    loss = cross_entropy_loss(logits=logits, labels=batch['label']) + loss_fair
    
    # if args.conf !='no_conf':
      
    #   if args.conf_method == "V": 
    #     loss += constraints_confidence(logits_fair)
    #   elif args.conf_method == "TV":
    #     loss += constraints_confidence(logits) + constraints_confidence(logits_fair)      
    #   else:
    #     raise NameError(f'Undefined conf_method. Should be TV or V. Current {args.conf_method}')

    return loss, (new_model_state, logits, logits_fair, lmd)

  return loss_fn



def get_loss_lmd_dynamic(state, batch, per_sample = False, T = None):
  args = global_var.get_value('args')
  mu = args.mu
  constraints_fair = constraints_dict[args.metric]
  constraints_confidence = constraints_dict[args.conf]
  def loss_fn(params, lmd): 
    if state.batch_stats:
        logits, new_model_state = state.apply_fn({'params': params, 'batch_stats': state.batch_stats}, batch['feature'], mutable=['batch_stats'])
    else:
        logits, new_model_state = state.apply_fn({'params': params}, batch['feature'], mutable=['batch_stats'])
    if len(logits) == 2: # logits and embeddings
        logits = logits[0]
    loss_reg, _ = constraints_fair(logits, batch['group'], batch['label'], T = T)
    lmd = lmd + mu * loss_reg 
    loss = cross_entropy_loss(logits=logits, labels=batch['label']) + jnp.sum(mu/2 * loss_reg**2) + jnp.sum(lmd * loss_reg)
    loss += constraints_confidence(logits)
    return loss, (new_model_state, logits, lmd)

  def loss_fn_per_sample(params, lmd): 
    if state.batch_stats:
        logits, new_model_state = state.apply_fn({'params': params, 'batch_stats': state.batch_stats}, batch['feature'], mutable=['batch_stats'])
    else:
        logits, new_model_state = state.apply_fn({'params': params}, batch['feature'], mutable=['batch_stats'])
    if len(logits) == 2: # logits and embeddings
        logits = logits[0]
    loss_reg, _ = constraints_fair(logits, batch['group'], batch['label']) # TODO: per sample
    lmd = lmd + mu * loss_reg 
    loss = cross_entropy_loss_per_sample(logits=logits, labels=batch['label']) + jnp.sum(mu/2 * loss_reg**2) + jnp.sum(lmd * loss_reg)
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
      logits, new_model_state = state.apply_fn({'params': params, 'batch_stats': state.batch_stats}, batch['feature'], mutable=['batch_stats'])
    else:
      logits, new_model_state = state.apply_fn({'params': params}, batch['feature'], mutable=['batch_stats'])
    if len(logits) == 2: # logits and embeddings
      logits, _ = logits
    else:
      raise NameError('Must have embeddings')

    # get weights
    weight_params = jax.tree_util.tree_leaves(params)
    weight_params_reg = jax.tree_util.tree_leaves(params_reg)
    weight_para_vec = (jnp.concatenate([x.reshape(-1) for x in weight_params]))
    weight_para_vec_reg = (jnp.concatenate([x.reshape(-1) for x in weight_params_reg]))

    loss_ce = cross_entropy_loss(logits=logits, labels=batch['label'])
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
      logits_main = state.apply_fn({'params': params, 'batch_stats': state.batch_stats}, batch['feature'], mutable=False, train=False)
    else:
      logits_main = state.apply_fn({'params': params}, batch['feature'], mutable=False, train=False)
    if len(logits_main) == 2: # logits and embeddings
      logits_main, _ = logits_main
    loss_main, aux_main = constraints_fair(logits_main, batch['group'], batch['label'])
    loss_conf_main = constraints_confidence_entropy(logits_main, batch['group'])
    # if len(logits) == 2: # logits and embeddings
    #   _, embeddings = logits
    # else:
    #   raise NameError('Must have embeddings')

    # if state_reg.batch_stats:
    #   logits, new_model_state_reg = state_reg.apply_fn({'params': params_reg, 'batch_stats': state_reg.batch_stats}, embeddings, mutable=['batch_stats'])
    # else:
    #   logits, new_model_state_reg = state_reg.apply_fn({'params': params_reg}, embeddings, mutable=['batch_stats'])

    if state_reg.batch_stats:
      logits, new_model_state_reg = state_reg.apply_fn({'params': params_reg, 'batch_stats': state_reg.batch_stats}, batch['feature'], mutable=['batch_stats'])
    else:
      logits, new_model_state_reg = state_reg.apply_fn({'params': params_reg}, batch['feature'], mutable=['batch_stats'])
    if len(logits) == 2: # logits and embeddings
      logits = logits[0]

    # get weights
    # weight_params = jax.tree_util.tree_leaves(params['head']) # only use the last linear layer
    weight_params = jax.tree_util.tree_leaves(params)
    weight_params_reg = jax.tree_util.tree_leaves(params_reg)
    weight_para_vec = (jnp.concatenate([x.reshape(-1) for x in weight_params]) )
    weight_para_vec_reg = (jnp.concatenate([x.reshape(-1) for x in weight_params_reg]) )


    loss_constraint, aux_reg = constraints_fair(logits, batch['group'], batch['label'])
    # loss_reg, aux_reg = constraints_dp_ranking(logits, batch['group'])
    # loss_reg, aux_reg = constraints_dp(logits, batch['group'])
    loss_conf = constraints_confidence_entropy(logits, batch['group'])
    # loss_two_net_gap = mu/2 * jnp.sum((weight_para_vec - weight_para_vec_reg + 1/mu * lmd)**2)

    loss_lmd = jnp.sum(lmd * (weight_para_vec - weight_para_vec_reg))
    loss_reg = mu/2 * jnp.sum(jnp.abs(weight_para_vec - weight_para_vec_reg)**2)
    # loss = jnp.abs(constraints_dp(logits, batch['group'])) + constraints_confidence(logits, batch['group']) + mu/2 * jnp.sum((weight_para_vec - weight_para_vec_reg + 1/mu * lmd)**2)
    loss =  jnp.abs(loss_constraint) + loss_lmd + loss_reg

    loss += constraints_confidence(logits)
    aux_reg_out = (aux_reg,loss_conf,jnp.abs(loss_constraint),loss_reg)
    aux_main_out = (aux_main,loss_conf_main,jnp.abs(loss_main),loss_reg)
    return loss, (new_model_state_reg, logits, lmd, (aux_reg_out,aux_main_out))
  return loss_fn