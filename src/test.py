# import jax
# import argparse
# # Options ----------------------------------------------------------------------
# parser = argparse.ArgumentParser()
# args = parser.parse_args()
# # data
# args.dataset = 'celeba'
# from jax.tree_util import tree_structure
# print(tree_structure(args.__dict__))

# def foo(**args):
#     return args
# args.dataset = 'celeba'

import numpy as np

a = np.array([[1,2,3],[4,5,6],[6,8,9],[10,11,12]])
print(a)
sel = np.array([0,1,2, 2])
# 
rnd_idx = np.arange(len(sel))
np.random.shuffle(rnd_idx)
num = a[range(len(a)),sel]
print(num - num[rnd_idx])