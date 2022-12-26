from flax import linen as nn
from jax import numpy as jnp
import jax
import optax
import numpy as np
import pdb



def cross_entropy_loss(logits, labels):
  return cross_entropy_loss_per_sample(logits, labels).mean()

def cross_entropy_loss_per_sample(logits, labels):
  if labels is None:
    return - jax.nn.log_softmax(logits, axis=-1)
  else:
    return optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=labels)

def hinge_loss(logits, labels):
  logits = logits.squeeze()
  return jnp.mean(jnp.maximum(1 - 2. * (labels.astype(jnp.float32) -0.5) * logits, 0))


def constraints_eod(logits, attributes, labels):
  EPS = 1e-8
  prob = jax.nn.softmax(logits) 

  group_a1 = jnp.sum((attributes == 0) * prob[:,1] * (labels == 1)) / jnp.sum( (labels == 1) * (attributes == 0) * 1.0 + EPS)
  group_b1 = jnp.sum((attributes == 1) * prob[:,1] * (labels == 1)) / jnp.sum( (labels == 1) * (attributes == 1) * 1.0 + EPS)

  group_a0 = jnp.sum((attributes == 0) * prob[:,1] * (labels == 0)) / jnp.sum( (labels == 0) * (attributes == 0) * 1.0 + EPS)
  group_b0 = jnp.sum((attributes == 1) * prob[:,1] * (labels == 0)) / jnp.sum( (labels == 0) * (attributes == 1) * 1.0 + EPS)

  return jnp.array([group_a0 - group_b0, group_a1 - group_b1]), (group_a1, group_b1)

def constraints_acc(logits, attributes, labels):
  EPS = 1e-8
  prob = jax.nn.softmax(logits) 

  # multi-class, multi-attributes
  num_groups = 3
  num_class = 10
  relaxed_acc_pergroup = []
  for group in range(num_groups):
    relaxed_acc_pergroup.append(jnp.sum((attributes == group) * prob[range(len(labels)), labels] ) / jnp.sum(  (attributes == group) * 1.0 + EPS))
  relaxed_acc_pergroup = jnp.array(relaxed_acc_pergroup)
  rnd_idx = jnp.arange(num_groups)
  jnp.random.shuffle(rnd_idx)

  loss_arr = relaxed_acc_pergroup - relaxed_acc_pergroup[rnd_idx]

  return loss_arr, relaxed_acc_pergroup

def constraints_eop(logits, attributes, labels):
  EPS = 1e-8
  prob = jax.nn.softmax(logits) 

  group_a1 = jnp.sum((attributes == 0) * prob[:,1] * (labels == 1)) / jnp.sum( (labels == 1) * (attributes == 0) * 1.0 + EPS)
  group_b1 = jnp.sum((attributes == 1) * prob[:,1] * (labels == 1)) / jnp.sum( (labels == 1) * (attributes == 1) * 1.0 + EPS)

  # group_a0 = jnp.sum((attributes == 0) * prob[:,1] * (labels == 0)) / jnp.sum( (labels == 0) * (attributes == 0) * 1.0 + EPS)
  # group_b0 = jnp.sum((attributes == 1) * prob[:,1] * (labels == 0)) / jnp.sum( (labels == 0) * (attributes == 1) * 1.0 + EPS)

  return group_a1 - group_b1, (group_a1, group_b1)

def constraints_dp_cov(logits, attributes, labels):
  """ E[f(X)Z] - E[f(X)]E[Z]
  """
  EPS = 1e-8
  prob = jax.nn.softmax(logits) 
  ez = jnp.mean(attributes)
  ef = jnp.mean(prob[:,1])
  efz = jnp.mean( attributes * prob[:,1] )

  # return efz - ez*ef, (efz, ez*ef)
  return (efz - ez*ef)*1.0, (prob[:,1], ef, ez, efz)


def constraints_eod_cov(logits, attributes, labels):
  prob = jax.nn.softmax(logits) 

  # loc0 = jnp.where(labels==0)
  # loc1 = jnp.where(labels==1)
  EPS = 1e-8
  ez0 = jnp.sum(attributes * (labels==0)*1.0) / (jnp.sum((labels==0)*1.0)   + EPS)
  ef0 = jnp.sum(prob[:,1] * (labels==0)*1.0) / (jnp.sum((labels==0)*1.0)   + EPS)
  efz0 = jnp.sum(prob[:,1] * attributes * ((labels==0)*1.0)) / (jnp.sum((labels==0)*1.0)   + EPS)


  ez1 = jnp.sum(attributes * (labels==1)*1.0) / (jnp.sum((labels==1)*1.0)   + EPS)
  ef1 = jnp.sum(prob[:,1] * (labels==1)*1.0) / (jnp.sum((labels==1)*1.0)   + EPS)
  efz1 = jnp.sum(prob[:,1] * attributes * ((labels==1)*1.0)) / (jnp.sum((labels==1)*1.0)   + EPS)

  # return efz - ez*ef, (efz, ez*ef)
  return jnp.array([(efz0 - ez0*ef0)*1.0,(efz1 - ez1*ef1)*1.0]), (prob[:,1], ef1, ez1, efz1)

def constraints_eop_cov(logits, attributes, labels):
  prob = jax.nn.softmax(logits) 


  EPS = 1e-8



  ez1 = jnp.sum(attributes * (labels==1)*1.0) / (jnp.sum((labels==1)*1.0)   + EPS)
  ef1 = jnp.sum(prob[:,1] * (labels==1)*1.0) / (jnp.sum((labels==1)*1.0)   + EPS)
  efz1 = jnp.sum(prob[:,1] * attributes * ((labels==1)*1.0)) / (jnp.sum((labels==1)*1.0)   + EPS)

  # return efz - ez*ef, (efz, ez*ef)
  return (efz1 - ez1*ef1)*1.0, (prob[:,1], ef1, ez1, efz1)

def constraints_dp_ranking(logits, attributes, labels):
  EPS = 1e-8
  prob = jax.nn.softmax(logits) # 
  # num_a, num_b = jnp.sum((attributes == 0) * 1.0), jnp.sum((attributes == 1) * 1.0)
  # pdb.set_trace()
  # min_num = min(num_a, num_b)
  # if min_num > 0:
  # group_a = jnp.where(attributes == 0, prob[:,1], 0)
  # group_b = jnp.where(attributes == 1, prob[:,1], 0)
  group_a = prob[len(prob)//2:,1]
  group_b = prob[:len(prob)//2,1]
  # group_a = prob[len(logits)//2:,1]
  # group_b = prob[:len(logits)//2,1]  

  loss_reg_detail = jnp.abs(jnp.sort(group_a) - jnp.sort(group_b))
  # return jnp.mean(jnp.sort(loss_reg_detail)[len(prob)//4*3:]), (jnp.sort(group_a), jnp.sort(group_b))
  # return jnp.mean(jnp.sort(loss_reg_detail)[100:]), (jnp.sort(group_a), jnp.sort(group_b))
  return jnp.sum(loss_reg_detail**2), (jnp.sort(group_a), jnp.sort(group_b))
  # else:
  #   group_a = jnp.sum((attributes == 0) * prob[:,1]) / jnp.sum((attributes == 0) * 1.0 + EPS)
  #   group_b = jnp.sum((attributes == 1) * prob[:,1]) / jnp.sum((attributes == 1) * 1.0 + EPS)
  #   return group_a - group_b 

def constraints_dp(logits, attributes, labels, T = None, M=2, K=2):
  EPS = 1e-8
  prob = jax.nn.softmax(logits)
  # M = len(np.unique(attributes))
  # K = len(jnp.unique(labels))
  constraint = []
  H_noisy = jnp.zeros((M,K))
  H_cal = jnp.zeros_like(H_noisy)
  for k in range(K):
    for i in range(M):
      H_noisy = H_noisy.at[i,k].set(jnp.sum((attributes == i) * prob[:,k]) / jnp.sum((attributes == i) * 1.0 + EPS)) 
    if T is not None:
      # pdb.set_trace()

      H_cal = H_cal.at[:,k].set(jnp.dot(T[k], H_noisy[:,k].reshape(-1,1)).reshape(-1))
      # H_cal[:,k] = jnp.matmul(T[k], H_noisy[:,k].reshape(-1,1))
    else:
      H_cal = H_cal.at[:,k].set(H_noisy[:,k])
      # H_cal[:,k] = H_noisy[:,k]
  H_cal_clip = jnp.clip(H_cal, 1e-3, 1)
  H_cal_clip_final = H_cal_clip / jnp.sum(H_cal_clip,1).reshape(-1,1)
  for i in range(M):
    for j in range(i+1, M):
      for k in range(K):
        constraint.append(H_cal_clip_final[i,k] - H_cal_clip_final[j,k])
  constraint = jnp.array(constraint) / (M*(M-1)*K*1.0)
  group_a = jnp.sum((attributes == 0) * prob[:,1]) / jnp.sum((attributes == 0) * 1.0 + EPS)
  group_b = jnp.sum((attributes == 1) * prob[:,1]) / jnp.sum((attributes == 1) * 1.0 + EPS)
  return constraint, (group_a, group_b)
  # T is not the original transition matrix. It has been transformed such that H[:,k] = T[k] H_noisy[:,k]
  # prob = jax.nn.softmax(prob) 
  # prob = logits
  # rev_prob = 1 - prob + EPS
  # group_a = jnp.where(attributes > 0, logits, 0).sum() / (jnp.where(attributes > 0, 1, 0).sum() + EPS)
  # group_b = jnp.where(attributes <= 0, logits, 0).sum() / (jnp.where(attributes <= 0, 1, 0).sum() + EPS)

  # group_a = jnp.sum((attributes == 0) * prob[:,1]) / jnp.sum((attributes == 0) * 1.0 + EPS)
  # group_b = jnp.sum((attributes == 1) * prob[:,1]) / jnp.sum((attributes == 1) * 1.0 + EPS)
  # return group_a - group_b, (group_a, group_b)

  # group_a2 = jnp.sum((attributes == 0) * prob[:,1]**2) / jnp.sum((attributes == 0) * 1.0 + EPS)
  # group_b2 = jnp.sum((attributes == 1) * prob[:,1]**2) / jnp.sum((attributes == 1) * 1.0 + EPS)

  # group_a3 = jnp.sum((attributes == 0) * prob[:,1]**3) / jnp.sum((attributes == 0) * 1.0 + EPS)
  # group_b3 = jnp.sum((attributes == 1) * prob[:,1]**3) / jnp.sum((attributes == 1) * 1.0 + EPS)


  # group_a2 = jnp.sum((attributes == 0) * jnp.log(prob[:,1]+EPS)) / jnp.sum((attributes == 0) * 1.0 + EPS)
  # group_b2 = jnp.sum((attributes == 1) * jnp.log(prob[:,1]+EPS)) / jnp.sum((attributes == 1) * 1.0 + EPS)


  # group_a3 = jnp.sum((attributes == 0) * jnp.log(1-prob[:,1]+EPS)) / jnp.sum((attributes == 0) * 1.0 + EPS)
  # group_b3 = jnp.sum((attributes == 1) * jnp.log(1-prob[:,1]+EPS)) / jnp.sum((attributes == 1) * 1.0 + EPS)
  # return jnp.abs(group_a - group_b) +  jnp.abs(group_a2 - group_b2) + jnp.abs(group_a3 - group_b3)

def constraints_confidence_entropy(logits):
  EPS = 1e-8
  prob = jax.nn.softmax(logits) + EPS
  confidence_loss = -jnp.mean(prob * jnp.log(prob))
  return confidence_loss

def constraints_confidence_peer(logits):
  EPS = 1e-8
  prob = jax.nn.softmax(logits) + EPS
  confidence_loss = jnp.mean(jnp.log(prob + EPS))
  return confidence_loss * 0.1

def constraints_confidence_no_conf(logits):
  return 0.0

def constraints_plain(logits, attributes, labels):
  return 0.0, 0.0



def correct(logits, labels):
  return jnp.argmax(logits, axis=-1) == jnp.argmax(labels, axis=-1)


def binary_correct(logits, labels):
  return (logits.squeeze() > 0) == labels


def accuracy(logits, labels):
  return jnp.mean(correct(logits, labels))


def fairness(logits, labels, attributes):
  # return jnp.where((attributes.squeeze() > 0) & (logits.squeeze() > 0), 1, 0).sum(), jnp.where((attributes.squeeze() == 0) & (logits.squeeze() > 0), 1, 0).sum()
  return jnp.where((attributes.squeeze() > 0) & (logits[:,0] > logits[:,1]), 1, 0).sum(), jnp.where((attributes.squeeze() == 0) & (logits[:,0] > logits[:,1]), 1, 0).sum()



def compute_metrics(logits, labels, groups):
  loss = cross_entropy_loss(logits=logits, labels=labels)
  accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
  ar0 = jnp.sum( (jnp.argmax(logits, -1) == 1) * (groups == 0) * 1.0) / jnp.sum( (groups == 0) * 1.0)
  ar1 = jnp.sum( (jnp.argmax(logits, -1) == 1) * (groups == 1) * 1.0) / jnp.sum( (groups == 1) * 1.0)

  tpr0 = jnp.sum( (jnp.argmax(logits, -1) == 1) * (groups == 0) * (labels==1) * 1.0) / jnp.sum( (groups == 0) * (labels==1) * 1.0)
  tpr1 = jnp.sum( (jnp.argmax(logits, -1) == 1) * (groups == 1) * (labels==1) * 1.0) / jnp.sum( (groups == 1) * (labels==1) * 1.0)

  fpr0 = jnp.sum( (jnp.argmax(logits, -1) == 1) * (groups == 0) * (labels==0) * 1.0) / jnp.sum( (groups == 0) * (labels==0) * 1.0)
  fpr1 = jnp.sum( (jnp.argmax(logits, -1) == 1) * (groups == 1) * (labels==0) * 1.0) / jnp.sum( (groups == 1) * (labels==0) * 1.0)

  # prob = jax.nn.softmax(logits)
  # ar0 = jnp.sum( (prob[:,1] == 1) * (groups == 0) * 1.0) / jnp.sum( (groups == 0) * 1.0) # only for controlled test
  # ar1 = jnp.sum( (prob[:,1] == 1) * (groups == 1) * 1.0) / jnp.sum( (groups == 1) * 1.0) # only for controlled test

  # op0 = jnp.sum( (jnp.argmax(logits, -1) == 1) * (labels == 1) * (groups == 0) * 1.0) / jnp.sum( (labels == 1) * (groups == 0) * 1.0)
  # op1 = jnp.sum( (jnp.argmax(logits, -1) == 1) * (labels == 1) * (groups == 1) * 1.0) / jnp.sum( (labels == 1) * (groups == 1) * 1.0)
  acc0 = jnp.sum( (jnp.argmax(logits, -1) == labels) * (groups == 0) * 1.0) / jnp.sum( (groups == 0) * 1.0)
  acc1 = jnp.sum( (jnp.argmax(logits, -1) == labels) * (groups == 1) * 1.0) / jnp.sum( (groups == 1) * 1.0)

  metrics = {
      'loss': loss,
      'accuracy': accuracy,
      'ar': (ar0, ar1),
      'acc': (acc0, acc1),
      'tpr': (tpr0, tpr1),
      'fpr': (fpr0, fpr1),
  }
  return metrics


constraints_dict = {
  'dp': constraints_dp,
  'dp_cov': constraints_dp_cov,
  'eop': constraints_eop,
  'eop_cov': constraints_eop_cov,
  'eod': constraints_eod,
  'eod_cov': constraints_eod_cov,
  'dp_ranking': constraints_dp_ranking,
  'plain': constraints_plain,
  'entropy': constraints_confidence_entropy,
  'peer': constraints_confidence_peer,
  'no_conf': constraints_confidence_no_conf,
}