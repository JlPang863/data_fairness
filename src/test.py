# import jax
# # # import argparse
# # # # Options ----------------------------------------------------------------------
# # # parser = argparse.ArgumentParser()
# # # args = parser.parse_args()
# # # # data
# # # args.dataset = 'celeba'
# # # from jax.tree_util import tree_structure
# # # print(tree_structure(args.__dict__))

# # # def foo(**args):
# # #     return args
# # # args.dataset = 'celeba'

# # import numpy as np

# # # a = np.array([[1,2,3],[4,5,6],[6,8,9],[10,11,12]])
# # # print(a)
# # # sel = np.array([0,1,2, 2])
# # # # 
# # # rnd_idx = np.arange(len(sel))
# # # np.random.shuffle(rnd_idx)
# # # num = a[range(len(a)),sel]
# # # print(num - num[rnd_idx])
# # # a = [1,3]
# # # a = np.array([-1,2,3,4,5])
# # a = [(1,2,[1],[2]), (1,2,[1,1],[2,2])]
# # # print(abs(a))
# # # a = {}
# # # a[(1,1)] = 1

# # np.save('result.npy', a)
# # b = np.load('result.npy', allow_pickle=True)
# # print(b)
# # # print(np.random.rand(np.sum(a>0)))
# import pickle


# with open('./src/recorder.pkl', 'rb') as f:
#     data = pickle.load(f)
# import pdb
# pdb.set_trace()

# a = [1,2,3,4]
# b = [-2:]

# a = set([1,2,3])
# b = a.copy()
# b.update([5])
# a = 1
# print(isinstance(a, int))
# import numpy as np
# print(np.random.choice(range(2), p = [0.5, 0.5]))

import torchvision
ds_2 = torchvision.datasets.ImageNet(root = '/data2/data', split='train', transform=None,
                                     target_transform=None, download=True)