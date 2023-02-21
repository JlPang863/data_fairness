# # import jax
# # # # import argparse
# # # # # Options ----------------------------------------------------------------------
# # # # parser = argparse.ArgumentParser()
# # # # args = parser.parse_args()
# # # # # data
# # # # args.dataset = 'celeba'
# # # # from jax.tree_util import tree_structure
# # # # print(tree_structure(args.__dict__))

# # # # def foo(**args):
# # # #     return args
# # # # args.dataset = 'celeba'

# # # import numpy as np

# # # # a = np.array([[1,2,3],[4,5,6],[6,8,9],[10,11,12]])
# # # # print(a)
# # # # sel = np.array([0,1,2, 2])
# # # # # 
# # # # rnd_idx = np.arange(len(sel))
# # # # np.random.shuffle(rnd_idx)
# # # # num = a[range(len(a)),sel]
# # # # print(num - num[rnd_idx])
# # # # a = [1,3]
# # # # a = np.array([-1,2,3,4,5])
# # # a = [(1,2,[1],[2]), (1,2,[1,1],[2,2])]
# # # # print(abs(a))
# # # # a = {}
# # # # a[(1,1)] = 1

# # # np.save('result.npy', a)
# # # b = np.load('result.npy', allow_pickle=True)
# # # print(b)
# # # # print(np.random.rand(np.sum(a>0)))
# # import pickle


# # with open('./src/recorder.pkl', 'rb') as f:
# #     data = pickle.load(f)
# # import pdb
# # pdb.set_trace()

# # a = [1,2,3,4]
# # b = [-2:]

# # a = set([1,2,3])
# # b = a.copy()
# # b.update([5])
# # a = 1
# # print(isinstance(a, int))
# # import numpy as np
# # print(np.random.choice(range(2), p = [0.5, 0.5]))

# import torchvision
# import torchvision.transforms as transforms
# import torch

# from typing import Any, Tuple
# class my_imagenet(torchvision.datasets.ImageNet):
#       def __getitem__(self, index: int) -> Tuple[Any, Any]:
#         """
#         Args:
#             index (int): Index

#         Returns:
#             tuple: (sample, target) where target is class_index of the target class.
#         """
#         path, target = self.samples[index]
#         sample = self.loader(path)
#         if self.transform is not None:
#             sample = self.transform(sample)
#         if self.target_transform is not None:
#             target = self.target_transform(target)

#         return sample, target, index


# train_transform = transforms.Compose([
#     transforms.Resize((32, 32)),
#     # transforms.RandomCrop(32, padding=4), 
#     # transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
# ])

# ds_new = my_imagenet(root = '/data2/data/imgnet/', split='train', transform=train_transform,
#                                      target_transform=None)

# # ds_new = torchvision.datasets.ImageNet(root = '/data2/data/imgnet/', split='train', transform=train_transform,
#                                     #  target_transform=None)


# dataloader_new = torch.utils.data.DataLoader(ds_new,
#                                         batch_size=min(len(ds_new), 128),
#                                         shuffle=True,
#                                         num_workers=1,
#                                         drop_last=True)

# new_iter = iter(dataloader_new)
# example = next(new_iter)

# import pdb
# pdb.set_trace()
print(set(None))