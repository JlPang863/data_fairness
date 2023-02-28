from src import train, train_general, global_var
from collections import OrderedDict
import argparse



# Options ----------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default='plain', help="plain fix_lmd dynamic_lmd")
parser.add_argument('--model', type=str, default='vit-b_8_lowres', help="resnet18_lowres vit-b_8_lowres")
parser.add_argument('--metric', type=str, default='dp', help="dp eop eod")
parser.add_argument('--lmd', type=float, default=0.0)
parser.add_argument('--tol', type=float, default=0.02) 
parser.add_argument('--without_label', default=False, action="store_true") # # get an unfair sample wp tol
parser.add_argument('--new_prob', type=float, default=0.9) 
parser.add_argument('--ratio_org', type=float, default=0.5) 
parser.add_argument('--aux_data', type=str, default=None, help="imagenet")
# parser.add_argument('--aux_data', nargs='+', type=str, default=None, help="imagenet scut self")
parser.add_argument('--half_ablation', default=False, action="store_true") # # get an unfair sample wp tol

parser.add_argument('--mu', type=float, default=1.0)
parser.add_argument('--warm_epoch', type=int, default=0)
parser.add_argument('--sel_layers', type=int, default=2)
parser.add_argument('--strategy', type=int, default=1)
parser.add_argument('--conf', type=str, default='no_conf', help='no_conf, peer, entropy')
parser.add_argument('--label_key', type=str, default='Smiling', help="5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young")

parser.add_argument('--label_ratio', type=float, default=0.01)
parser.add_argument('--val_ratio', type=float, default=0.1)

# Example: CUDA_VISIBLE_DEVICES=0 python3 run_celeba.py --method plain  --warm_epoch 0  --metric dp --label_ratio 0.05 --val_ratio 0.1 --strategy 2 



# setup
ROOT = '.'
EXP = 'exps'
RUN = 0
META_MODEL_SEED, META_TRAIN_SEED, SEED_INCR = 42, 4242, 424242
EP_STEPS = 200  # 200
DATA_DIR = '/data2/data'
EXPS_DIR = ROOT + '/exps'

# arguments
# args = SimpleNamespace()
args = parser.parse_args()
# data
args.data_dir = DATA_DIR
args.dataset = 'celeba'

# model
# args.model = 'resnet18_lowres' 
# args.model = 'vit-b_8_lowers'
args.model_seed = META_MODEL_SEED + RUN * SEED_INCR


# optimizer
args.lr = 0.01
args.momentum = 0.9
args.weight_decay = 0.0005
args.nesterov = True
# SGD
args.opt = OrderedDict(
    name="sgd",
    config=OrderedDict(
        learning_rate = args.lr,
        momentum = args.momentum,
        nesterov = args.nesterov
    )
)
args.scheduler = None



# training
args.num_epochs = 10 + args.warm_epoch
args.EP_STEPS = EP_STEPS
args.train_seed = META_TRAIN_SEED + RUN * SEED_INCR
args.train_batch_size = 256 # 256
args.test_batch_size = 4096
# checkpoints
args.log_steps = EP_STEPS
args.save_steps =  EP_STEPS


# experiment
args.datasize = 202599
args.num_classes = 2
args.attr_key = 1
args.feature_key = 0
args.idx_key = 2
args.group_key = "Male"
args.img_size = 32
args.balance_batch = False
args.new_data_each_round = args.train_batch_size # 1024


args.train_conf = False
args.remove_pos = True
args.remove_posOrg = False


args.save_dir = EXPS_DIR + f'/{EXP}/{args.method}/run_{RUN}_{args.label_key}_warm{args.warm_epoch}_metric_{args.metric}'
if __name__ == "__main__":

    # data conversion for torch loader
    attributes_names = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']
    
    args.group_key = attributes_names.index(args.group_key)
    args.label_key = attributes_names.index(args.label_key)

    if 'resnet' in args.model:
        args.sel_layers = args.sel_layers
    elif 'vit' in args.model:
        args.sel_layers = -args.sel_layers
    global_var.init()
    global_var.set_value('args', args)
    # train(args)
    train_general(args)

