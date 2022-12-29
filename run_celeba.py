from src import train, global_var
from collections import OrderedDict
import argparse



# Options ----------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default='plain', help="plain fix_lmd dynamic_lmd admm")
parser.add_argument('--fe_sel', type=int, default=0, help="0--5")
parser.add_argument('--metric', type=str, default='dp', help="dp eop eod")
# parser.add_argument('--model_sel', type=int, default=1, help="VGG-Face, Facenet Facenet512,OpenFace, DeepFace, DeepID, ArcFace, Dlib, SFace")
# parser.add_argument('--e1', type=float, default=0.0)
# parser.add_argument('--e2', type=float, default=0.0)
parser.add_argument('--lmd', type=float, default=1.0)
parser.add_argument('--tol', type=float, default=0.0) # # get an unfair sample wp tol

parser.add_argument('--mu', type=float, default=1.0)
parser.add_argument('--warm_epoch', type=int, default=1)
parser.add_argument('--strategy', type=int, default=1)
parser.add_argument('--conf', type=str, default='false', help='no_conf, peer, entropy')
parser.add_argument('--label_key', type=str, default='Smiling', help="5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young")

parser.add_argument('--label_ratio', type=float, default=0.01)
parser.add_argument('--val_ratio', type=float, default=0.1)

# Example: LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 python3 run_celeba.py --method dynamic_lmd  --lmd 0.0 --mu 1.0  --warm_epoch 0 --metric dp --conf entropy
# setup
ROOT = '.'
EXP = 'exps'
RUN = 0
META_MODEL_SEED, META_TRAIN_SEED, SEED_INCR = 42, 4242, 424242
EP_STEPS = 200  # 200
# DATA_DIR = ROOT + './fair-eval/celeba/data'
# DATA_DIR = ROOT + '/data'
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
args.model = 'vit-b_8'
args.model_seed = META_MODEL_SEED + RUN * SEED_INCR
args.load_dir = None
args.ckpt = 0

# optimizer
args.lr = 0.01
args.momentum = 0.9
args.weight_decay = 0.0005
args.nesterov = True
# args.lr_vitaly = False
# args.decay_factor = 0.2
# args.decay_steps = [50*EP_STEPS, 80*EP_STEPS, 90*EP_STEPS]
# Adam
# args.opt = OrderedDict(
#     name="adam",
#     config=OrderedDict(
#         learning_rate = 0.01
#     )
# )
# SGD
args.opt = OrderedDict(
    name="sgd",
    config=OrderedDict(
        learning_rate = args.lr,
        momentum = args.momentum,
        nesterov = args.nesterov
    )
)
# cosine scheduler
# args.scheduler = None
args.scheduler = OrderedDict(
    name = "cosine_decay_schedule",
    config = OrderedDict(
        init_value = args.lr,
        decay_steps = 5000,  # previous: 5000, 10 epochs
        alpha = 0.95,
    )
)

# training
args.num_epochs = 50
args.EP_STEPS = EP_STEPS
args.train_seed = META_TRAIN_SEED + RUN * SEED_INCR
args.train_batch_size = 256
# args.train_batch_size = 64
args.test_batch_size = 4096
args.augment = True
args.track_forgetting = True
# checkpoints
args.log_steps = EP_STEPS
args.save_steps =  EP_STEPS


# experiment
args.datasize = 202599
args.num_classes = 2
# args.attr_key = "attributes"
# args.feature_key = "image"
args.attr_key = 1
args.feature_key = 0
args.idx_key = 2
# args.label_key = "Smiling" 
# args.label_key = "Attractive" 
args.group_key = "Male"
args.img_size = 32
args.balance_batch = False
args.new_data_each_round = 128
args.sampling_rounds = args.num_epochs * 2


method_list = [
        "Facenet", 
        "Facenet512", 
        "OpenFace", 
        "ArcFace", 
        "Dlib", 
        "SFace",
        ]
if args.fe_sel == 6:
    args.feature_extractor = 'None'
else:
    args.feature_extractor = method_list[args.fe_sel]
args.save_dir = EXPS_DIR + f'/{EXP}/{args.method}/run_{RUN}_{args.label_key}_lmd{args.lmd}_mu{args.mu}_warm{args.warm_epoch}_metric_{args.metric}_conf_{args.conf}_{args.feature_extractor}'
if __name__ == "__main__":

    # data conversion for torch loader
    attributes_names = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']
    
    args.group_key = attributes_names.index(args.group_key)
    args.label_key = attributes_names.index(args.label_key)


    global_var.init()
    global_var.set_value('args', args)
    train(args) # disparity mitigation with our method

