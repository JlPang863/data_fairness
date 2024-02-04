from src import train, train_jigsaw, global_var
from collections import OrderedDict
import argparse

#train.py
import jax
# import tensorflow as tf
from jax import numpy as jnp
import numpy as np
import time
from .data import   load_data, gen_preprocess_func_torch2jax
from .models import get_model
from .recorder import init_recorder, record_train_stats, save_recorder, record_test, save_checkpoint
import pdb
from .hoc_fairlearn import *
from .train_state import test_step, get_train_step, create_train_state, infl_step, infl_step_fair, infl_step_per_sample
from .metrics import compute_metrics, compute_metrics_fair
from .utils import set_global_seed, make_dirs, log_and_save_args
from . import global_var
import collections
import os





# bayesian active learning's repo
from baal.active import get_heuristic
from baal.active.active_loop import ActiveLearningLoop
from baal.bayesian.dropout import patch_module


from torch import nn
from torch import optim
from torch.nn import CrossEntropyLoss
from tqdm import tqdm


# Options ----------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default='plain', help="plain fix_lmd dynamic_lmd")
parser.add_argument('--model', type=str, default='mlp', help="mlp")
parser.add_argument('--metric', type=str, default='dp', help="dp eop eod")
parser.add_argument('--lmd', type=float, default=0.0)
parser.add_argument('--tol', type=float, default=0.0) # # get an unfair sample wp tol

parser.add_argument('--mu', type=float, default=1.0)
parser.add_argument('--warm_epoch', type=int, default=0)
parser.add_argument('--sel_layers', type=int, default=4)
parser.add_argument('--strategy', type=int, default=1)
parser.add_argument('--conf', type=str, default='no_conf', help='no_conf, peer, entropy')
parser.add_argument('--aux_data', type=str, default=None, help="imagenet")


parser.add_argument('--label_ratio', type=float, default=0.1)
parser.add_argument('--val_ratio', type=float, default=0.2)

#new add arguments for testing
parser.add_argument('--new_prob', type=float, default=0.5) 
parser.add_argument('--ratio_org', type=float, default=0.5) 




## baseline specific metric

parser.add_argument("--dataset")
parser.add_argument("--attribute")
parser.add_argument("--target_key")
parser.add_argument("--random", default=1337)
parser.add_argument("--oracle", action='store_true')
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--initial_pool", default=500, type=int)
parser.add_argument("--query_size", default=100, type=int)
parser.add_argument("--lr", default=0.01)
parser.add_argument("--heuristic", default="random", type=str)
parser.add_argument("--iterations", default=20, type=int)
parser.add_argument("--lambda", default=0, type=float)
parser.add_argument('--learning_epoch', default=10, type=int)
parser.add_argument('--weight_decay', default=0, type=float)


# Example: CUDA_VISIBLE_DEVICES=0 python3 run_celeba.py --method plain  --warm_epoch 0  --metric dp --label_ratio 0.05 --val_ratio 0.1 --strategy 2 



# setup
ROOT = '.'
EXP = 'exps'
RUN = 0
META_MODEL_SEED, META_TRAIN_SEED, SEED_INCR = 42, 4242, 424242
EP_STEPS = 1000  # 200
DATA_DIR = '/data2/data'
EXPS_DIR = ROOT + '/exps'

# arguments
# args = SimpleNamespace()
args = parser.parse_args()
# data
args.data_dir = DATA_DIR
args.dataset = 'jigsaw'

# model
args.model_seed = META_MODEL_SEED + RUN * SEED_INCR


# optimizer
args.lr = 0.0001
args.momentum = 0.9
args.weight_decay = 0.0005
args.nesterov = True
# SGD
# args.opt = OrderedDict(
#     name="sgd",
#     config=OrderedDict(
#         learning_rate = args.lr,
#         momentum = args.momentum,
#         nesterov = args.nesterov
#     )
# )
args.opt = OrderedDict(
    name="adam",
    config=OrderedDict(
        learning_rate = args.lr,
        # momentum = args.momentum,
        # nesterov = args.nesterov
    )
)
args.scheduler = None



# training
args.num_epochs = 10 + args.warm_epoch
args.EP_STEPS = EP_STEPS
args.train_seed = META_TRAIN_SEED + RUN * SEED_INCR
args.train_batch_size = 64
args.test_batch_size = 4096
# checkpoints
args.log_steps = EP_STEPS
args.save_steps =  EP_STEPS


# experiment
# args.datasize = 202599
args.num_classes = 2
args.balance_batch = False
args.new_data_each_round = 128 # 1024


args.train_conf = False
args.remove_pos = True
args.remove_posOrg = False


args.save_dir = EXPS_DIR + f'/{EXP}/{args.method}/run_{RUN}_warm{args.warm_epoch}_metric_{args.metric}'


    # baseline 1: BALD, which using ActiveLearningLoop to label data
    # elif args.strategy == 6:
    #   #MC dropoout
    #   heuristic = get_heuristic('random')
    #   loop_oracle = heuristic
    #   active_loop = ActiveLearningLoop(example,
    #                                   model.predict_on_dataset_generator,
    #                                   loop_oracle,
    #                                   100,
    #                                   batch_size=6,
    #                                   iterations=20,
    #                                   use_cuda= torch.cuda.is_available(),
    #                                   workers=0)


import torch
from baal import ModelWrapper
from baal.utils.cuda_utils import to_cuda
from pytorch_revgrad import RevGrad
from torch import nn


class GradCrit(nn.Module):
    def __init__(self, crit, lmd, attribute):
        super().__init__()
        self.crit = crit
        self.lmb = lmd
        self.attribute = attribute

    def forward(self, input, target):
        if self.training:
            cls_pred, group_pred = input
            cls_loss = self.crit(cls_pred, target['target'])
            group_loss = self.crit(group_pred, target[self.attribute])
            return cls_loss + self.lmb * group_loss
        else:
            return self.crit(input, target['target'])


class GRADWrapper(ModelWrapper):
    def predict_on_batch(self, data, iterations=1, cuda=False):
        out = super().predict_on_batch(data, iterations, cuda)
        # Return clss
        return out[0]

    def train_on_batch(self, data, target, optimizer, cuda=False,
                       regularizer=None):
        if cuda:
            data, target = to_cuda(data), to_cuda(target)
        optimizer.zero_grad()
        output = self.model(data)
        loss = self.criterion(output, target)

        if regularizer:
            regularized_loss = loss + regularizer()
            regularized_loss.backward()
        else:
            loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
        optimizer.step()
        self._update_metrics(output, target, loss, filter='train')
        return loss


class GRADModel(nn.Module):
    def __init__(self, model, num_groups):
        super().__init__()
        self.model = model
        self.num_groups = num_groups
        self.group_pred = nn.Sequential(
            RevGrad(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_groups)
        )

    def forward(self, x):
        x = self.model.features(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x1 = self.model.classifier(x)
        x2 = self.group_pred(x)
        return x1, x2



def train_bald(args):

    '''
    load data
    '''
    # setup
    set_global_seed(args.train_seed)
    # make_dirs(args)
    train_loader_labeled, train_loader_unlabeled, idx_with_labels = load_data(args, args.dataset, mode = 'train', aux_dataset=None)

    # retrieve data from dataloader
    labeled_dataset = train_loader_labeled.dataset.dataset
    unlabled_dataset = train_loader_unlabeled.dataset.dataset

    #train set
    active_set = np.concatenate(labeled_dataset, unlabled_dataset)

    

    #validation set / test set
    val_loader, test_loader = load_data(args, args.dataset, mode = 'val')
    val_set = val_loader.dataset.dataset
    test_set = test_loader.dataset.dataset


    #preprocess data 
    preprocess_func_torch2jax = gen_preprocess_func_torch2jax(args.dataset)
    active_set = preprocess_func_torch2jax(active_set, args)
    val_set = preprocess_func_torch2jax(val_set, args)
    test_set = preprocess_func_torch2jax(test_set, args)



    num_classes = 2
    num_group = 2

    '''
    model settings
    '''
    #################################################################################
    # model setting
    heuristic = get_heuristic(args.heuristic)
    criterion = GradCrit(CrossEntropyLoss(), lmd=0, attribute=attribute)


    #model = utils.vgg16(pretrained=True, num_classes=num_classes)
    model = get_model(args)

    # create a GRADModel
    model = GRADModel(model, num_group)

    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    # change dropout layer to MCD
    model = patch_module(model)

    #################################################################################

    '''
    specific setting: fairness metric
    '''
    # redefine model again
    model = GRADWrapper(model, criterion)
    # model.add_metric('fair_recall',
    #                  lambda: FairnessMetric(skm.recall_score, 'recall', average='micro',
    #                                         attribute=attribute))
    # model.add_metric('fair_accuracy',
    #                  lambda: FairnessMetric(skm.accuracy_score, 'accuracy',
    #                                         attribute=hyperparams['attribute']))
    # model.add_metric('fair_precision',
    #                  lambda: FairnessMetric(skm.precision_score, 'precision', average='micro',
    #                                         attribute=attribute))
    # model.add_metric('fair_f1',
    #                  lambda: FairnessMetric(skm.f1_score, 'f1', average='micro',
    #                                         attribute=attribute))

    # save imagenet weights
    init_weights = deepcopy(model.state_dict())

    # for prediction we use a smaller batchsize
    # since it is slower
    if hyperparams['oracle']:
        loop_oracle = BalancedHeuristic(active_set, heuristic, hyperparams['attribute'])
    else:
        loop_oracle = heuristic


    #labeling data samples
    active_loop = ActiveLearningLoop(active_set,
                                     model.predict_on_dataset_generator,
                                     loop_oracle,
                                     hyperparams['query_size'],
                                     batch_size=6,
                                     iterations=20,
                                     use_cuda=use_cuda,
                                     workers=0)
    learning_epoch = hyperparams['learning_epoch']



    # ######################### starting training ############################
    for epoch in tqdm(itertools.count(start=0), desc="Active loop"):
        if len(active_set) > 20000:
            break

        # set to train mode
        criterion.train()
        model.load_state_dict(init_weights)

        # train step
        model.train_on_dataset(active_set, optimizer, hyperparams["batch_size"],
                               learning_epoch, use_cuda, workers=0)

        # Validation!
        #set to valuation model
        criterion.eval()

        #test step
        model.test_on_dataset(test_set, batch_size=6, use_cuda=use_cuda, workers=0,
                              average_predictions=hyperparams['iterations'])
        fair_logs = {}


        for met in ['fair_recall', 'fair_accuracy', 'fair_precision', 'fair_f1']:
            fair_test = model.metrics[f'test_{met}'].value
            fair_test = {'test_' + k: v for k, v in fair_test.items()}
            fair_train = model.metrics[f'train_{met}'].value
            fair_train = {'train_' + k: v for k, v in fair_train.items()}
            fair_logs.update(fair_test)
            fair_logs.update(fair_train)
        metrics = model.metrics

        should_continue = active_loop.step()
        if not should_continue:
            break


        # Send logs
        train_loss = metrics['train_loss'].value
        val_loss = metrics['test_loss'].value

        logs = {
            "test_loss": val_loss,
            "train_loss": train_loss,
            "epoch": epoch,
            "labeled_data": active_set.labelled,
            "next_training_size": len(active_set)
        }
        logs.update(fair_logs)
        print(logs)

if __name__ == "__main__":


    # if 'mlp' in args.model:
    #     args.sel_layers = -args.sel_layers
    global_var.init()
    global_var.set_value('args', args)
    #train(args)
    train_bald(args)

