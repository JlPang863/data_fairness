# from . import global_var
import numpy as np
# import jax.numpy as jnp
# import jax
import pdb
import torchvision
import torchvision.transforms as transforms
import torch
import random
from typing import Any, Tuple
import PIL
import os
from .utils import preprocess_compas, race_encode
import sklearn.preprocessing as preprocessing

def preprocess_func_compas_torch(example, args, noisy_attribute = None, num_groups = 2):
  """ preprocess the data
  """

  feature, group, label = example[0].numpy(), example[2].numpy().astype(np.uint8), example[1].numpy().astype(np.uint8)
  group[group >= num_groups] = num_groups - 1

  # use str to avoid error in Jax tree
  # args.feature_key, args.label_key, args.group_key = f'{args.feature_key}', f'{args.label_key}', f'{args.group_key}' 
  
  if noisy_attribute is None:
    data = {
      'feature': feature,
      'label': label,
      'group': group,
      'index': example[3].numpy()
    }
  else:
    noisy_attribute = noisy_attribute[:,0]
    data = {
      'feature': feature,
      'label': label,
      'group': noisy_attribute,
      'index': example[3].numpy()
    }
    # print(np.mean((noisy_attribute==group)*1.0))
  # global_var.set_value('args', args)
  return data



def preprocess_func_imgnet_torch(example, args, noisy_attribute = None, new_labels = {}):
  """ preprocess the data
  """

  image, group, label, idx = example[0].numpy(), None, example[1].numpy().astype(np.uint8),  example[2].numpy()

  image = image.transpose((0, 2, 3, 1)) 

  if len(new_labels) > 0:
    label = np.asarray([new_labels[idx[i]] if idx[i] in new_labels else label[i]  for i in range(len(idx))])
  if noisy_attribute is None:
    data = {
      'feature': image,
      'label': label,
      'group': group,
      'index': idx
    }
  else:
    noisy_attribute = noisy_attribute[:,0]
    data = {
      'feature': image,
      'label': label,
      'group': noisy_attribute,
      'index': idx
    }

  return data


def preprocess_func_celeba_torch(example, args, noisy_attribute = None, new_labels = {}):
  """ preprocess the data
  """

  image, group, label = example[args.feature_key].numpy(), example[args.attr_key][:,args.group_key].numpy().astype(np.uint8), example[args.attr_key][:,args.label_key].numpy().astype(np.uint8)
  idx = example[args.idx_key].numpy()
  pdb.set_trace()
  image = image.transpose((0, 2, 3, 1)) 
  # use str to avoid error in Jax tree
  # args.feature_key, args.label_key, args.group_key = f'{args.feature_key}', f'{args.label_key}', f'{args.group_key}' 
  
  if len(new_labels) > 0:
    label = np.asarray([new_labels[idx[i]] if idx[i] in new_labels else label[i]  for i in range(len(idx))])
  if noisy_attribute is None:
    data = {
      'feature': image,
      'label': label,
      'group': group,
      'index': idx
    }
  else:
    noisy_attribute = noisy_attribute[:,0]
    data = {
      'feature': image,
      'label': label,
      'group': noisy_attribute,
      'index': idx
    }
    # print(np.mean((noisy_attribute==group)*1.0))
  # global_var.set_value('args', args)
  return data

def gen_preprocess_func_torch2jax(dataset):
  if dataset == 'celeba':
    return preprocess_func_celeba_torch
  elif dataset == 'compas':
    return preprocess_func_compas_torch
  elif dataset == 'imagenet':
    return preprocess_func_imgnet_torch
  else:
    return None






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




class CompasDataset(torch.utils.data.Dataset):
  


  def __init__(self, data_file, args, split = 'train'):
      FEATURES_CLASSIFICATION = ["age_cat", "sex", "priors_count", "c_charge_degree", 'decile_score', 'length_of_stay'] #features to be used for classification
      CONT_VARIABLES = ["priors_count", 'decile_score', 'length_of_stay'] # continuous features, will need to be handled separately from categorical features, categorical features will be encoded using one-hot
      CLASS_FEATURE = "two_year_recid" # the decision variable


      self.df = data_file.copy()
      data = data_file.to_dict('list')
      for k in data.keys():
          data[k] = np.array(data[k])


      Y = data[CLASS_FEATURE]
  
      X = np.array([]).reshape(len(Y), 0) # empty array with num rows same as num examples, will hstack the features to it

      feature_names = []
      # import pdb
      # pdb.set_trace()
      for attr in FEATURES_CLASSIFICATION:
          vals = data[attr]
          if attr in CONT_VARIABLES:
              vals = [float(v) for v in vals]
              vals = preprocessing.scale(vals) # 0 mean and 1 variance
              vals = np.reshape(vals, (len(Y), -1)) # convert from 1-d arr to a 2-d arr with one col
              # pdb.set_trace()

          else: # for binary categorical variables, the label binarizer uses just one var instead of two
              lb = preprocessing.LabelBinarizer()
              lb.fit(vals)
              vals = lb.transform(vals)
              # pdb.set_trace()


          # add to learnable features
          X = np.hstack((X, vals))

          if attr in CONT_VARIABLES: # continuous feature, just append the name
              feature_names.append(attr)
          else: # categorical features
              if vals.shape[1] == 1: # binary features that passed through lib binarizer
                  feature_names.append(attr)
              else:
                  for k in lb.classes_: # non-binary categorical features, need to add the names for each cat
                      feature_names.append(attr + "_" + str(k))

      
      self.feature = torch.tensor(X, dtype=torch.float)
      self.label = torch.tensor(Y, dtype=torch.long)
      self.true_attribute = data['race']       
      self.score = torch.tensor(self.df.decile_score.to_list(), dtype=torch.long).view(-1,1)

      idx = list(range(len(self.label)))
      random.Random(args.train_seed).shuffle(idx)
      num = int(len(self.label) * 0.8)
      if split == 'train':
        idx = idx[:num].copy()
      else:
        idx = idx[num:].copy()
      idx = np.asarray(idx)
      self.feature = self.feature[idx]
      self.label = self.label[idx]
      self.true_attribute = self.true_attribute[idx]  
      self.score = self.score[idx]

      print(f'dataset construction done. \nShape of X {self.feature.shape}. \nShape of Y {self.label.shape}')
    
  def __len__(self):
      return len(self.label)

  def __getitem__(self, index):
      feature, label, group = self.feature[index], self.label[index], self.true_attribute[index]
      return feature, label, group, index


class my_imagenet(torchvision.datasets.ImageNet):
      def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, index

def load_celeba_dataset_torch(args, shuffle_files=False, split='train', batch_size=128, ratio = 0.1, sampled_idx = None, return_part2 = False, fair_train=False, aux_dataset = None):

  train_transform = transforms.Compose([
      transforms.Resize(args.img_size),
      # transforms.RandomCrop(32, padding=4), 
      # transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
  ])

  test_transform = transforms.Compose([
      transforms.Resize(args.img_size),
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
  ])

  args.input_shape = (1, args.img_size, args.img_size, 3)
  if split == 'train':
    transform = train_transform
  else:
    transform = test_transform

  ds = my_celeba(root = args.data_dir, split = split, target_type = 'attr', transform = transform, download = True)
  if split == 'train':
    args.datasize = len(ds)
    

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
    # part1 += sampled_idx
    ds_new = torch.utils.data.Subset(ds, sampled_idx)
    part2 = list(set(part2) - set(sampled_idx))
  
    print(f'{len(part1)} originally labeled samples, {len(sampled_idx)} new samples, and {len(part2)} unlabeled samples. Total: {len(part1) + len(part2) + len(sampled_idx)}')
  else:
    print(f'{len(part1)} originally labeled samples, 0 new samples, and {len(part2)} unlabeled samples. Total: {len(part1) + len(part2)}')



  ds_1 = torch.utils.data.Subset(ds, part1)
  if len(part2) > 0:
    ds_2 = torch.utils.data.Subset(ds, part2)
  else:
    ds_2 = torch.utils.data.Subset(ds, part1) # just a placeholder

  if aux_dataset == 'imagenet':
    ds_2 = my_imagenet(root = args.data_dir + '/imgnet/', split='train', transform=train_transform,
                                     target_transform=None)
  else:
    if aux_dataset is not None:
      raise NameError(f'Undefined dataset {aux_dataset}')

  if fair_train:
    dataloader_1 = torch.utils.data.DataLoader(ds_1,
                                              batch_size=batch_size if split == 'train' else 256, # val loader: 256
                                              shuffle=shuffle_files,
                                              num_workers=1,
                                              drop_last=True)
    dataloader_2 = torch.utils.data.DataLoader(ds_2,
                                            batch_size=batch_size,
                                            shuffle=shuffle_files,
                                            num_workers=1,
                                            drop_last=False)
  else:
    dataloader_1 = torch.utils.data.DataLoader(ds_1,
                                              batch_size=batch_size if split == 'train' else 512, # val loader: 512
                                              shuffle=shuffle_files,
                                              num_workers=1,
                                              drop_last=True)
    dataloader_2 = torch.utils.data.DataLoader(ds_2,
                                            batch_size=batch_size if split == 'test' else 32, # unlabeled loader: 32
                                            shuffle=shuffle_files,
                                            num_workers=1,
                                            drop_last=False)
    if sampled_idx is not None:
      dataloader_new = torch.utils.data.DataLoader(ds_new,
                                              batch_size=min(len(ds_new), batch_size),
                                              shuffle=shuffle_files,
                                              num_workers=1,
                                              drop_last=True)



  if return_part2:
    return [dataloader_1, dataloader_2], part1, part2
  else:
    if sampled_idx is not None:
      return [dataloader_1, dataloader_2, dataloader_new], part1
    else:
      return [dataloader_1, dataloader_2], part1




def load_compas_dataset_torch(args, shuffle_files=False, split='train', batch_size=128, ratio = 0.1, sampled_idx = None, return_part2 = False, fair_train=False):

  

  # get compas data
  df = preprocess_compas()
  race_encode(df)
  ds = CompasDataset(df.copy(), args, split=split)

  args.input_shape = (1, ds.feature.shape[1])
  if split == 'train':
    args.datasize = len(ds)


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
  if len(part2) > 0:
    ds_2 = torch.utils.data.Subset(ds, part2)
  else:
    ds_2 = torch.utils.data.Subset(ds, part1) # just a placeholder

  

  if fair_train:
    dataloader_1 = torch.utils.data.DataLoader(ds_1,
                                              batch_size=batch_size if split == 'train' else 256, # val loader: 256
                                              shuffle=shuffle_files,
                                              num_workers=1,
                                              drop_last=True)
    dataloader_2 = torch.utils.data.DataLoader(ds_2,
                                            batch_size=batch_size,
                                            shuffle=shuffle_files,
                                            num_workers=1,
                                            drop_last=False)
  else:
    dataloader_1 = torch.utils.data.DataLoader(ds_1,
                                              batch_size=batch_size if split == 'train' else 512, # val loader: 512
                                              shuffle=shuffle_files,
                                              num_workers=1,
                                              drop_last=False)
    dataloader_2 = torch.utils.data.DataLoader(ds_2,
                                            batch_size=batch_size if split == 'test' else 32, # unlabeled loader: 32
                                            shuffle=shuffle_files,
                                            num_workers=1,
                                            drop_last=False)

  if return_part2:
    return [dataloader_1, dataloader_2], part1, part2
  else:
    return [dataloader_1, dataloader_2], part1

def load_jigsaw_dataset(args, mode='train', ratio=0.1):
    path = args.data_dir  # 'data/'
    if mode == 'train':
      with open(os.path.join(path, 'Jigsaw/Jigsaw_train.npy'), 'rb') as f:
        X, Y, A = np.load(f), np.load(f), np.load(f)
        # shuffle training data
        from sklearn import utils
        X, Y, A = utils.shuffle(X, Y, A)
    else:
      with open(os.path.join(path, 'Jigsaw/Jigsaw_test.npy'), 'rb') as f:
        X, Y, A = np.load(f), np.load(f), np.load(f)

    index = np.arange(X.shape[0])
    labeled_index, unlabeled_index = np.split(index, [ratio * X.shape[0]])
    
    encode_data = lambda X, Y, A, I: {"feature":X, "label":Y, 'group':A, 'index':I}

    labeled_data = encode_data(X[labeled_index], Y[labeled_index], A[labeled_index], labeled_index)
    unlabeled_data = encode_data(X[unlabeled_index], Y[unlabeled_index], A[unlabeled_index], unlabeled_index)

    return labeled_data, unlabeled_data

def load_data(args, dataset, mode = 'train', sampled_idx = None, aux_dataset = None):
  
  if dataset == 'celeba':
    if mode == 'train':
      
      
      if sampled_idx is not None:
        [train_loader_labeled, train_loader_unlabeled, train_loader_new], part_1 = load_celeba_dataset_torch(args, shuffle_files=True, split='train', batch_size=args.train_batch_size, ratio = args.label_ratio, sampled_idx=sampled_idx, aux_dataset = aux_dataset)
        idx_with_labels = set(part_1)
        return train_loader_labeled, train_loader_unlabeled, train_loader_new, idx_with_labels
      else:
        [train_loader_labeled, train_loader_unlabeled], part_1 = load_celeba_dataset_torch(args, shuffle_files=True, split='train', batch_size=args.train_batch_size, ratio = args.label_ratio, sampled_idx=sampled_idx, aux_dataset = aux_dataset)
        idx_with_labels = set(part_1)
        return train_loader_labeled, train_loader_unlabeled, idx_with_labels

    elif mode == 'val':
      [val_loader, test_loader], _ = load_celeba_dataset_torch(args, shuffle_files=True, split='test', batch_size=args.test_batch_size, ratio = args.val_ratio)
      return val_loader, test_loader
    else:
      raise NotImplementedError('mode should be either train or val')
  elif dataset == 'compas':
    if mode == 'train': # TODO
      [train_loader_labeled, train_loader_unlabeled], part_1 = load_compas_dataset_torch(args, shuffle_files=True, split='train', batch_size=args.train_batch_size, ratio = args.label_ratio, sampled_idx=sampled_idx)
      idx_with_labels = set(part_1)
      return train_loader_labeled, train_loader_unlabeled, idx_with_labels
    elif mode == 'val':
      [val_loader, test_loader], _ = load_compas_dataset_torch(args, shuffle_files=True, split='test', batch_size=args.test_batch_size, ratio = args.val_ratio)
      return val_loader, test_loader
    else:
      raise NotImplementedError('mode should be either train or val')


