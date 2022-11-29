from . import global_var
import numpy as np
import jax.numpy as jnp
import jax
import pdb
import torchvision
import torchvision.transforms as transforms
import torch

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
  args.feature_key, args.label_key, args.group_key = f'{args.feature_key}', f'{args.label_key}', f'{args.group_key}' 
  
  if noisy_attribute is None:
    data = {
      args.feature_key: image,
      args.label_key: label,
      args.group_key: group,
      'index': example[args.idx_key].numpy()
    }
  else:
    noisy_attribute = noisy_attribute[:,0]
    data = {
      args.feature_key: image,
      args.label_key: label,
      args.group_key: noisy_attribute,
      'index': example[args.idx_key].numpy()
    }
    # print(np.mean((noisy_attribute==group)*1.0))
  global_var.set_value('args', args)
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
              # TODO: refactor with utils.verify_str_arg
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


def load_celeba_dataset_torch(args, shuffle_files=False, split='train', batch_size=128):

  train_transform = transforms.Compose([
      transforms.Resize(32),
      transforms.RandomCrop(32, padding=4), 
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0)),
  ])

  test_transform = transforms.Compose([
      transforms.Resize(32),
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0)),
  ])

  if split == 'train':
    transform = train_transform
  else:
    transform = test_transform

  ds = my_celeba(root = args.data_dir, split = split, target_type = 'attr', transform = transform, download = True)
  dataloader = torch.utils.data.DataLoader(ds,
                                            batch_size=batch_size,
                                            shuffle=shuffle_files,
                                            num_workers=4,
                                            drop_last=False)

  return dataloader
