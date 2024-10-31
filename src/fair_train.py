
import jax
# import tensorflow as tf
import numpy as np
import time
import os
from .data import  load_data, gen_preprocess_func_torch2jax
from .models import get_model
from .recorder import init_recorder, record_train_stats, record_test, save_checkpoint,load_checkpoint
from .hoc_fairlearn import *
from .train_state import test_step, get_train_step, create_train_state
from .metrics import compute_metrics_fair
from .utils import set_global_seed, log_and_save_args
import logging
from .loss_func import *
from typing import Any,Callable
from scipy.special import xlogy

from jax import jacrev, numpy as jnp
from jax.flatten_util import ravel_pytree
from jax import jacfwd, jacrev, jit, vmap
from jax.tree_util import tree_flatten, tree_map


os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # Disables GPU memory preallocation, allowing JAX to allocate as needed.

def test(args, state, data):
    """Runs the test step and computes fair metrics."""
    logits, labels, groups = [], [], []
    preprocess_func = gen_preprocess_func_torch2jax(args.dataset)
    
    for example in data:
        batch = preprocess_func(example, args, noisy_attribute=None)
        logit = test_step(state, batch)
        logits.append(logit)
        labels.append(batch['label'])
        groups.append(batch['group'])

    return compute_metrics_fair(
        logits=jnp.concatenate(logits),
        labels=jnp.concatenate(labels),
        groups=jnp.concatenate(groups),
    )

def fair_train_validation(args):
    """Main function for fair training validation."""
    set_global_seed(args.train_seed)
    preprocess_func = gen_preprocess_func_torch2jax(args.dataset)

    # Load data
    train_loader_labeled, train_loader_unlabeled, idx_with_labels = load_data(args, args.dataset, mode='train', aux_dataset=args.aux_data)
    _, train_loader_unlabeled_org, idx_with_labels_org = load_data(args, args.dataset, mode='train', aux_dataset=None)
    args.train_with_org = True
    val_loader, test_loader = load_data(args, args.dataset, mode='val')

    # Load model from checkpoint if available
    load_checkpoint_model = True
    if load_checkpoint_model:
        checkpoint = load_checkpoint(args.save_dir + '/ckpts')
        model = get_model(args)
        state = create_train_state(model, args, checkpoint['params'], return_opt=False)
    else:
        model = get_model(args)
        state = create_train_state(model, args, return_opt=False)

    # Model size information
    num_layers = len(jax.tree_util.tree_leaves(state.params))
    print(f'Number of layers: {num_layers}')

    # Initialize recorder and training configuration
    rec = init_recorder()
    log_and_save_args(args)
    time_start = time.time()
    lmd = args.lmd
    train_step, train_step_warm = get_train_step(args.method)
    worst_group_id = 0

    # Begin training loop
    for epoch_i in range(args.num_epochs):
        args.curr_epoch = epoch_i
        t, num_sample_cur = 0, 0
        print(f'Epoch {epoch_i}')
        val_iter = iter(val_loader)
        args.datasize = len(val_loader.dataset)

        while t * args.train_batch_size < args.datasize:
            for example in val_loader:  # Training on validation data for fairness
                try:
                    example_fair = next(val_iter)
                except StopIteration:
                    val_iter = iter(val_loader)
                    example_fair = next(val_iter)

                bsz = example[0].shape[0]
                num_sample_cur += bsz
                batch = preprocess_func(example, args, noisy_attribute=None)
                batch_fair = preprocess_func(example_fair, args, noisy_attribute=None)
                t += 1

                if t * args.train_batch_size > args.datasize:
                    break

                # Training step based on method
                if args.method == 'plain':
                    state, train_metric = train_step(state, batch)
                elif args.method == 'dynamic_lmd' and state.step >= args.warm_step:
                    state, train_metric, train_metric_fair, lmd = train_step(state, batch, batch_fair, lmd=lmd, T=None, worst_group_id=worst_group_id)
                else:
                    state, train_metric, _, lmd = train_step_warm(state, batch, batch_fair, lmd=lmd, T=None, worst_group_id=worst_group_id)

                rec = record_train_stats(rec, t-1, train_metric, 0)

                if t % args.log_steps == 0:
                    test_metric = test(args, state, test_loader)
                    val_metric = test(args, state, val_loader)
                    worst_group_id = np.argmin(val_metric['acc'])
                    rec, _ = record_test(rec, t + args.datasize * epoch_i // args.train_batch_size, args.datasize * args.num_epochs // args.train_batch_size, time.time(), time_start, train_metric, test_metric)

                    print(f'lmd: {lmd}')

        rec = save_checkpoint(args.save_dir, t + args.datasize * epoch_i // args.train_batch_size, state, rec, save=False)
