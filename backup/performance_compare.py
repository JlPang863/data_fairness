import pickle
import numpy as np
import pdb
from src import func_cvx_bound
import matplotlib.pyplot as plt
import argparse

# Options ----------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--fair_metric', type=str, default='dp', help="dp eop eod")
parser.add_argument('--label_key', type=str, default='Smiling', help="5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young")
args = parser.parse_args()

confs = ['entropy', 'no_conf', 'peer']
# metrics = ['dp', 'dp_cov', 'plain']
fair_metric = args.fair_metric
metrics = [fair_metric, fair_metric+'_cov', 'plain']
# warm = [0, 1]
# confs = ['entropy']
# metrics = ['dp']
# warm = [0]
# methods = ['dynamic_lmd']
# lmds = [0.0]
RUN = 0
mu = 1.0
method = 'dynamic_lmd'
lmd = 0.0
warm_epoch = 1
# for method in methods:
#     for lmd in lmds:
#         for warm_epoch in warm:
# file_name = './exps/exps/dynamic_lmd/run_0_lmd0.0_mu1.0_warm0_metric_dp_conf_entropy/recorder.pkl'
# plt.figure()
fig, axs = plt.subplots(len(metrics), len(confs), figsize=(10,10))

for metric_i in range(len(metrics)):
    for conf_j in range(len(confs)):
        file_name = f'./exps/exps/{method}/run_{RUN}_{args.label_key}_lmd{lmd}_mu{mu}_warm{warm_epoch}_metric_{metrics[metric_i]}_conf_{confs[conf_j]}/recorder.pkl'
        try:
            print(file_name)
            with open(file_name, 'rb') as f:
                data = pickle.load(f)
                
                ax = axs[metric_i,conf_j]
                acc = np.array(data['test_acc'])
                if fair_metric == 'dp':
                    ar = np.array(data['test_ar'])
                    ar_gap = np.abs(ar[:,0] - ar[:,1])
                    ar_avg = np.mean(ar,1)
                elif fair_metric == 'eod':
                    # ar = np.vstack((np.array(data['test_tpr']),np.array(data['test_fpr'])))
                    ar1 = np.array(data['test_tpr'])
                    ar2 = np.array(data['test_fpr'])
                    ar_gap = np.abs(ar1[:,0] - ar1[:,1]) + np.abs(ar2[:,0] - ar2[:,1])
                    ar_avg = np.mean(ar1,1)/2 + np.mean(ar2,1)/2
                    # pdb.set_trace()
                elif fair_metric == 'eop':
                    ar = np.array(data['test_tpr'])
                    ar_gap = np.abs(ar[:,0] - ar[:,1])
                    ar_avg = np.mean(ar,1)
                
                
                points = np.vstack((ar_gap,acc)).transpose()
                
                vertices_bound = func_cvx_bound(points)
                # pdb.set_trace()
                ax.plot(points[:,0], points[:,1], '.')
                ax.plot(points[vertices_bound,0], points[vertices_bound,1], 'r--', lw=2)
                ax.plot(points[vertices_bound[0],0], points[vertices_bound[0],1], 'ro')
                ax.set_title(f'metric_{metrics[metric_i]}_conf_{confs[conf_j]}')
                ax.set_xlabel(f'fairness gap')
                ax.set_ylabel(f'test acc')
                # ax.set_ylim([0.82,0.9])
                # ax.set_xlim([0.00,0.15])
                # plt.show()
                # exit()

                
        except:
            print(f'file {file_name} not found')

save_name = f'./results/{args.label_key}_{method}_{fair_metric}_lmd{lmd}_mu{mu}_warm{warm_epoch}.pdf'
plt.tight_layout()
plt.savefig(save_name)
# plt.show()
# print(data)
# import pdb

# pdb.set_trace()

