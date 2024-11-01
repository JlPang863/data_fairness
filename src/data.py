import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch
import random
from typing import Any, Tuple
import PIL
import os
from .utils import preprocess_compas, race_encode, preprocess_adult, preprocess_jigsaw
import sklearn.preprocessing as preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sklearn.preprocessing as preprocessing
import tensorflow as tf
import tensorflow_datasets as tfds
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
from collections import Counter

def preprocess_func_compas_torch(example, args, noisy_attribute = None, num_groups = 2):
  """ preprocess the data
  """

  feature, group, label = example[0].numpy(), example[2].numpy().astype(np.uint8), example[1].numpy().astype(np.uint8)
  

  for i in range(len(group)):
    if group[i] >=1:
      group[i] = 1
    else:
      group[i] = 0

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
  return data

def preprocess_func_celeba_torch(example, args, noisy_attribute = None, new_labels = {}):
  """ preprocess the data
  """

  image, group, label = example[args.feature_key].numpy(), example[args.attr_key][:,args.group_key].numpy().astype(np.uint8), example[args.attr_key][:,args.label_key].numpy().astype(np.uint8)
  idx = example[args.idx_key].numpy()

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

def preprocess_func_adult_torch(example, args, noisy_attribute = None, num_groups = 2):
  """ preprocess the data
  """

  feature, group, label = example[0].numpy(), example[2].numpy().astype(np.uint8), example[1].numpy().astype(np.uint8)

  ## change to binary case
  if args.group_key == 'race':
    for i in range(len(group)):
      if group[i] > 3:
        group[i] = 1
      else:
        group[i] = 0
  elif args.group_key == 'age':
    for i in range(len(group)):
      if group[i] >= 35: 
        group[i] = 1
      else:
        group[i] = 0



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
  return data


def preprocess_func_jigsaw_torch(example, args, noisy_attribute = None, num_groups = 2):
  """ preprocess the data
  """

  feature, group, label = example[0].numpy(), example[2].numpy().astype(np.uint8), example[1].numpy().astype(np.uint8)
  group[group >= num_groups] = num_groups - 1

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
  return data

def gen_preprocess_func_torch2jax(dataset):
  if dataset == 'celeba':
    return preprocess_func_celeba_torch
  elif dataset == 'compas':
    return preprocess_func_compas_torch
  elif dataset == 'adult':
     return preprocess_func_adult_torch
  elif dataset == 'jigsaw':
     return preprocess_func_jigsaw_torch
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

      for attr in FEATURES_CLASSIFICATION:
          vals = data[attr]
          if attr in CONT_VARIABLES:
              vals = [float(v) for v in vals]
              vals = preprocessing.scale(vals) # 0 mean and 1 variance
              vals = np.reshape(vals, (len(Y), -1)) # convert from 1-d arr to a 2-d arr with one col

          else: # for binary categorical variables, the label binarizer uses just one var instead of two
              lb = preprocessing.LabelBinarizer()
              lb.fit(vals)
              vals = lb.transform(vals)


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


class my_scut(torch.utils.data.Dataset):

    def __init__(self, root, transform):
        data_dir = root + '/train_test_files/All_labels.txt'
        with open(data_dir, 'r') as f:
            lines = f.readlines()  
            path = []   
            label = [] 
            for line in lines:
                linesplit = line.split('\n')[0].split(' ')
                addr = linesplit[0]
                target = torch.Tensor([float(linesplit[1])])
                path.append(addr)
                if target >= 3.0:
                    label.append(1)
                else:
                    label.append(0)
        self.path = path
        self.label = label
        self.transform = transform
        self.root = root


    def __getitem__(self, index):
        sample = PIL.Image.open(os.path.join(self.root, "Images", self.path[index]))
        target = self.label[index]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target, index

    def __len__(self):
        return len(self.path)



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

class AdultDataset(torch.utils.data.Dataset):
    def __init__(self, df, args, split = 'train'):
        self.features = df.drop(columns=["income"]).values
        self.labels = df["income"].values
        if args.group_key == 'sex':
          self.groups = df["sex"].values
        elif args.group_key == 'race':
          self.groups = df['race'].values
        elif args.group_key == 'age':
           self.groups = df['age'].values
        else:
           raise NameError('unknow group key!')
        idx = list(range(len(self.labels)))


        
        random.Random(args.train_seed).shuffle(idx)
        num = int(len(self.labels) * 0.8)


        # use 80% adult data for train, 20% for test
        if split == 'train':
          idx = idx[:num].copy()
        else:
          idx = idx[num:].copy()
        idx = np.asarray(idx)
        self.features = self.features[idx]
        self.labels = self.labels[idx]
        self.groups = self.groups[idx]


        print(f'dataset construction done. \n Shape of X {self.features.shape}. \nShape of Y {self.labels.shape}')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float), torch.tensor(self.labels[idx], dtype=torch.long), torch.tensor(self.groups[idx], dtype=torch.long), idx

class JigsawDataset(torch.utils.data.Dataset):
    def __init__(self, data_dict):
        self.features = data_dict["feature"]
        self.labels  = data_dict['label']
        self.groups = data_dict['group']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float), torch.tensor(self.labels[idx], dtype=torch.long), torch.tensor(self.groups[idx], dtype=torch.long), idx


import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler

class InverseProportionalSampler(WeightedRandomSampler):
    def __init__(self, labels, replacement=True):
        unique_labels, counts = np.unique(labels, return_counts=True)
      
        proportions = counts / len(labels)
        
        label_to_weight = {label: 1.0/proportion for label, proportion in zip(unique_labels, proportions)}
        
        label_to_weight = {
            0: 1,
            1: 6
        }
        weights = np.array([label_to_weight[label] for label in labels])
        super(InverseProportionalSampler, self).__init__(weights, len(labels), replacement)

class InverseProportionalSampler_Adult(WeightedRandomSampler):
    def __init__(self, labels, replacement=True):
        unique_labels, counts = np.unique(labels, return_counts=True)
        proportions = counts / len(labels)
        
        label_to_weight = {label: 1.0/proportion for label, proportion in zip(unique_labels, proportions)}

        label_to_weight = {
            0: 1,
            1: 1, 
        }
        weights = np.array([label_to_weight[label] for label in labels])
        super(InverseProportionalSampler_Adult, self).__init__(weights, len(labels), replacement)


def get_labels_for_subset(dataset, indices):
    labels = []
    for idx in indices:
        _, label, _ = dataset[idx]  
        labels.append(label)
    return labels




def load_celeba_dataset_torch(args, shuffle_files=False, split='train', batch_size=128, ratio = 0.1, sampled_idx = [], return_part2 = False, fair_train=False, aux_dataset = None):


  train_transform = transforms.Compose([
      transforms.Resize((int(218 / 178 * args.img_size), args.img_size)),
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
  ])

  test_transform = transforms.Compose([
      transforms.Resize((int(218 / 178 * args.img_size), args.img_size)),
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
  ])

  args.input_shape = (1, int(218 / 178 * args.img_size), args.img_size, 3)
  if split == 'train':
    transform = train_transform
  else:
    transform = test_transform

  ds = my_celeba(root = args.data_dir, target_type = 'attr', transform = transform, download = True)
  
  args.datasize = len(ds)
  
  # data split
  # train --> train_labeled (1) + train_unlabeled (2), ratio is for train_labeled
  # test --> val (1) + test (2), ratio is for val
  full_index = list(range(len(ds)))
  train_size = int(len(ds)*0.8) #split train test dataset

  if split == 'train':
    idx = full_index[:train_size]

  elif split == 'test':
    idx = full_index[train_size:]


  random.Random(args.train_seed).shuffle(idx)
  num = int(len(ds) * ratio)
  part1 = idx[:num]
  part2 = idx[num:]

  if len(sampled_idx) > 0:
    ds_new = torch.utils.data.Subset(ds, sampled_idx)
    part2 = list(set(part2) - set(sampled_idx))
  
    print(f'{len(part1)} originally labeled samples, {len(sampled_idx)} new samples, and {len(part2)} unlabeled samples. Total: {len(part1) + len(part2) + len(sampled_idx)}')
  else:
    print(f'{len(part1)} originally labeled samples, 0 new samples, and {len(part2)} unlabeled samples. Total: {len(part1) + len(part2)}')

  if len(part2) > 0:
    ds_2 = torch.utils.data.Subset(ds, part2)
  else:
    ds_2 = torch.utils.data.Subset(ds, part1) 


  ds_1 = torch.utils.data.Subset(ds, part1)


  if fair_train:
    dataloader_1 = torch.utils.data.DataLoader(ds_1,
                                              batch_size=batch_size if split == 'train' else 256, 
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
                                              batch_size=batch_size if split == 'train' else 256, 
                                              shuffle=shuffle_files,
                                              num_workers=1,
                                              drop_last=True)
    dataloader_2 = torch.utils.data.DataLoader(ds_2,
                                            batch_size=batch_size if split == 'test' else 32, 
                                            shuffle=shuffle_files,
                                            num_workers=1,
                                            drop_last=False)
    if len(sampled_idx) > 0:
      dataloader_new = torch.utils.data.DataLoader(ds_new,
                                              batch_size=min(len(ds_new), batch_size),
                                              shuffle=shuffle_files,
                                              num_workers=1,
                                              drop_last=True
                                                )



  if return_part2:
    return [dataloader_1, dataloader_2], part1, part2
  else:
    if len(sampled_idx) > 0:
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
                                              batch_size=batch_size if split == 'train' else 256, 
                                              shuffle=shuffle_files,
                                              num_workers=0,
                                              drop_last=True)
    dataloader_2 = torch.utils.data.DataLoader(ds_2,
                                            batch_size=batch_size,
                                            shuffle=shuffle_files,
                                            num_workers=0,
                                            drop_last=False)
  else:
    dataloader_1 = torch.utils.data.DataLoader(ds_1,
                                              batch_size=batch_size if split == 'train' else 512, 
                                              shuffle=shuffle_files,
                                              num_workers=0,
                                              drop_last=False)
    dataloader_2 = torch.utils.data.DataLoader(ds_2,
                                            batch_size=batch_size if split == 'test' else 32, 
                                            shuffle=shuffle_files,
                                            num_workers=0,
                                            drop_last=False)

  if return_part2:
    return [dataloader_1, dataloader_2], part1, part2
  else:
    return [dataloader_1, dataloader_2], part1


def load_jigsaw_dataset_torch(args, shuffle_files=False, split='train', batch_size=128, ratio=0.1, sampled_idx=None, return_part2=False):
  path = '/data1/jialu/'
  if split == 'train':
    with open(os.path.join(path, 'Jigsaw_train.npy'), 'rb') as f:
      X, Y, A = np.load(f), np.load(f), np.load(f)

      from sklearn import utils
      X, Y, A = utils.shuffle(X, Y, A, random_state=args.train_seed)

  else:
    with open(os.path.join(path, 'Jigsaw_test.npy'), 'rb') as f:
      X, Y, A = np.load(f), np.load(f), np.load(f)


  args.input_shape = (1, X.shape[1])

  args.datasize = X.shape[0]
  index = np.arange(X.shape[0])
  random.Random(args.train_seed).shuffle(index)

  split_point = int(ratio * X.shape[0]) 

  part1, part2 = index[:split_point], index[split_point:]
  
  if sampled_idx is not None:
      part1 = part1.tolist()
      part1 +=  sampled_idx
      part2 = list(set(part2) - set(sampled_idx))

  print(f'{len(part1)} labeled samples and {len(part2)} unlabeled samples. Total: {len(part1) + len(part2)}')

  encode_data = lambda X, Y, A, I: {"feature":X, "label":Y, 'group':A, 'index':I}
  
  labeled_dataset_index = encode_data(X[part1], Y[part1], A[part1], part1)
  if len(part2) > 0: 
    unlabeled_dataset_index = encode_data(X[part2], Y[part2], A[part2], part2)
  else:
    unlabeled_dataset_index = encode_data(X[part1], Y[part1], A[part1], part1)

  labeled_dataset = JigsawDataset(labeled_dataset_index)
  unlabeled_dataset = JigsawDataset(unlabeled_dataset_index)
  
  # Create PyTorch Dataloaders
  labels_1 = labeled_dataset.labels.tolist()
  labels_2 = unlabeled_dataset.labels.tolist()

  # for label balance
  if split == 'train':
    sampler_1 = InverseProportionalSampler(labels_1)
    sampler_2 = InverseProportionalSampler(labels_2)

    dataloader_1 = DataLoader(labeled_dataset, batch_size=batch_size,  num_workers=0, drop_last=False, sampler = sampler_1)
    dataloader_2 = DataLoader(unlabeled_dataset, batch_size=batch_size, num_workers=0, drop_last=False,  sampler = sampler_2)
  else:
    dataloader_1 = DataLoader(labeled_dataset, batch_size=batch_size, shuffle=shuffle_files,  num_workers=0, drop_last=False)
    dataloader_2 = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=shuffle_files, num_workers=0, drop_last=False)



  if return_part2:
      return [dataloader_1, dataloader_2], part1, part2
  else:
      return [dataloader_1, dataloader_2], part1


def load_adult_dataset_torch(args, shuffle_files=False, split='train', batch_size=128, ratio=0.1, sampled_idx=None, return_part2=False):
    # Get adult data
    df = preprocess_adult()
    
    ds = AdultDataset(df,args, split)

    args.input_shape = (1, ds.features.shape[1])
    if split == 'train':
        args.datasize = len(ds)

    # Data split
    idx = list(range(len(ds)))
    random.Random(args.train_seed).shuffle(idx)
    num = int(len(ds) * ratio)
    part1 = idx[:num]
    part2 = idx[num:]

    if sampled_idx is not None:
        part1 += sampled_idx
        part2 = list(set(part2) - set(sampled_idx))
    print(f'{len(part1)} labeled samples and {len(part2)} unlabeled samples. Total: {len(part1) + len(part2)}')

    ds_1 = Subset(ds, part1)
    ds_2 = Subset(ds, part2)


    # Create a random subset sampler for training
    labels_ds_1 = [ds.labels[i] for i in part1]
    labels_ds_2 = [ds.labels[i] for i in part2]

    # for label balance
    sampler_1 = InverseProportionalSampler_Adult(labels_ds_1)
    sampler_2 = InverseProportionalSampler_Adult(labels_ds_2)
    if split == 'train':
      # for balance batch's labels
      dataloader_1 = DataLoader(ds_1, batch_size=batch_size,  num_workers=0, drop_last=False, sampler = sampler_1)
      dataloader_2 = DataLoader(ds_2, batch_size=batch_size, num_workers=0, drop_last=False,  sampler = sampler_2)
    else:
      dataloader_1 = DataLoader(ds_1, batch_size=batch_size, shuffle=shuffle_files,  num_workers=0, drop_last=False)
      dataloader_2 = DataLoader(ds_2, batch_size=batch_size, shuffle=shuffle_files,  num_workers=0, drop_last=False)

    if return_part2:
        return [dataloader_1, dataloader_2], part1, part2
    else:
        return [dataloader_1, dataloader_2], part1




def load_data(args, dataset, mode = 'train', sampled_idx = [], aux_dataset = None):
  
  if dataset == 'celeba':
    if mode == 'train':
      if len(sampled_idx) > 0:
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
    if mode == 'train': 
      [train_loader_labeled, train_loader_unlabeled], part_1 = load_compas_dataset_torch(args, shuffle_files=True, split='train', batch_size=args.train_batch_size, ratio = args.label_ratio, sampled_idx=sampled_idx)
      idx_with_labels = set(part_1)
      return train_loader_labeled, train_loader_unlabeled, idx_with_labels
    elif mode == 'val':
      [val_loader, test_loader], _ = load_compas_dataset_torch(args, shuffle_files=True, split='test', batch_size=args.test_batch_size, ratio = args.val_ratio)
      return val_loader, test_loader
    
  elif dataset == 'adult':
    if mode == 'train': 
      [train_loader_labeled, train_loader_unlabeled], part_1 = load_adult_dataset_torch(args, shuffle_files=True, split='train', batch_size=args.train_batch_size, ratio = args.label_ratio, sampled_idx=sampled_idx)
      idx_with_labels = set(part_1)
      return train_loader_labeled, train_loader_unlabeled, idx_with_labels
    elif mode == 'val':
      [val_loader, test_loader], _ = load_adult_dataset_torch(args, shuffle_files=True, split='test', batch_size=args.test_batch_size, ratio = args.val_ratio)
      return val_loader, test_loader

  elif dataset == 'jigsaw':
    if mode == 'train': 
      [train_loader_labeled, train_loader_unlabeled], part_1 = load_jigsaw_dataset_torch(args, shuffle_files=True, split='train', batch_size=args.train_batch_size, ratio = args.label_ratio, sampled_idx=sampled_idx)
      idx_with_labels = set(part_1)
      return train_loader_labeled, train_loader_unlabeled, idx_with_labels
    elif mode == 'val':
      [val_loader, test_loader], _ = load_jigsaw_dataset_torch(args, shuffle_files=True, split='test', batch_size=args.test_batch_size, ratio = args.val_ratio)
      return val_loader, test_loader
  else:
    raise NameError('Unamed dataset!')


