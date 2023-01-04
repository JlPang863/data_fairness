from . import global_var
import numpy as np
import jax.numpy as jnp
import jax
import pdb
import torchvision
import torchvision.transforms as transforms
import torch
import random
from typing import Any, Tuple
import PIL
import os



def preprocess_func_celeba_torch(example, args, noisy_attribute = None):
  """ preprocess the data
  """
  # ATTR_KEY = "attributes"
  # IMAGE_KEY = "image"
  # LABEL_KEY = "Smiling"
  # GROUP_KEY = "Male"


  image, group, label = example[args.feature_key].numpy(), example[args.attr_key][:,args.group_key].numpy().astype(np.uint8), example[args.attr_key][:,args.label_key].numpy().astype(np.uint8)
  # pdb.set_trace()
  image = image.transpose((0, 2, 3, 1)) 
  # use str to avoid error in Jax tree
  # args.feature_key, args.label_key, args.group_key = f'{args.feature_key}', f'{args.label_key}', f'{args.group_key}' 
  
  if noisy_attribute is None:
    data = {
      'feature': image,
      'label': label,
      'group': group,
      'index': example[args.idx_key].numpy()
    }
  else:
    noisy_attribute = noisy_attribute[:,0]
    data = {
      'feature': image,
      'label': label,
      'group': noisy_attribute,
      'index': example[args.idx_key].numpy()
    }
    # print(np.mean((noisy_attribute==group)*1.0))
  # global_var.set_value('args', args)
  return data



class my_celeba(torchvision.datasets.CelebA):

  def __getitem__(self, index: int) -> Tuple[Any, Any]:
      X = PIL.Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", self.filename[index]))

      target: Any = []
      for t in self.target_type:
          if t == "attr":
              target.append(self.attr[index, :])
          elif t == "identity":
              target.append(self.identity[index, 0])
          elif t == "bbox":
              target.append(self.bbox[index, :])
          elif t == "landmarks":
              target.append(self.landmarks_align[index, :])
          else:
              raise ValueError(f'Target type "{t}" is not recognized.')

      if self.transform is not None:
          X = self.transform(X)

      if target:
          target = tuple(target) if len(target) > 1 else target[0]

          if self.target_transform is not None:
              target = self.target_transform(target)
      else:
          target = None

      return X, target, index


def load_celeba_dataset_torch(args, shuffle_files=False, split='train', batch_size=128, ratio = 0.1, sampled_idx = None):

  train_transform = transforms.Compose([
      transforms.Resize(32),
      # transforms.RandomCrop(32, padding=4), 
      # transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
  ])

  test_transform = transforms.Compose([
      transforms.Resize(32),
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
  ])

  if split == 'train':
    transform = train_transform
  else:
    transform = test_transform

  ds = my_celeba(root = args.data_dir, split = split, target_type = 'attr', transform = transform, download = True)

  # data split
  # train --> train_labeled (1) + train_unlabeled (2), ratio is for train_labeled
  # test --> val (1) + test (2), ratio is for val
  # if split == 'train':
  idx = list(range(len(ds)))
  random.Random(args.train_seed).shuffle(idx)
  num = int(len(ds) * ratio)
  part1 = idx[:num]
  part2 = idx[num:]

  if sampled_idx is not None:
    part1 += sampled_idx
    part2 = list(set(part2) - set(sampled_idx))
  print(f'{len(part1)} labeled samples and {len(part2)} unlabeled samples. Total: {len(part1) + len(part2)}')


  ds_1 = torch.utils.data.Subset(ds, part1)
  ds_2 = torch.utils.data.Subset(ds, part2)


  dataloader_1 = torch.utils.data.DataLoader(ds_1,
                                            batch_size=batch_size if split == 'train' else 128, # val loader: 512
                                            shuffle=shuffle_files,
                                            num_workers=4,
                                            drop_last=False)
  dataloader_2 = torch.utils.data.DataLoader(ds_2,
                                          batch_size=batch_size if split == 'test' else 8, # unlabeled loader: 32
                                          shuffle=shuffle_files,
                                          num_workers=4,
                                          drop_last=False)

  return [dataloader_1, dataloader_2], part1
