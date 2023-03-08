import numpy as np
import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt


class Gaussian_dataset(torch.utils.data.Dataset):

    def __init__(self, candidate_size = 2000 * 2, train_size = 100, val_size = 200, test_size = 1000, seed = 0):
        num_bins = 30
        np.random.seed(seed)
        
        
        
        self.build_subset(candidate_size, 'candidate')
        self.build_subset(train_size, 'train')
        self.build_subset(val_size, 'val')
        self.build_subset(test_size, 'test')


        


        # build biased dataset


        # plt.hist(all_data_0, bins = np.linspace(-5,5,num_bins))
        # plt.hist(all_data_1, bins = np.linspace(-5,5,num_bins))
        # plt.show()


    def build_subset(self, num_samples, name):
        scale = 1
        shift = 1

        # build sub-dataset
        if name == 'train':
            biased = True
        else:
            biased = False

        print(f'Build {name} dataset ...')
        all_data_0 = np.random.randn(num_samples) * scale + shift
        all_data_1 = np.random.randn(num_samples) * scale - shift
        candidate_data = np.hstack((all_data_0, all_data_1))
        label_0 = [-1] * num_samples
        label_1 = [1] * num_samples
        label = np.asarray(label_0 + label_1)
        group = np.asarray([0] * (num_samples // 2) + [1] * (num_samples // 2) + [0] * (num_samples // 2) + [1] * (num_samples // 2))
        candidate_dataset = np.stack((candidate_data, label, group), axis = 1)
        if biased:
            idx = np.arange(len(candidate_dataset))[(candidate_dataset[:,0] > 1) * (candidate_dataset[:,2] == 1)]
            candidate_dataset = np.delete(candidate_dataset, idx, axis=0)

        print(f'Build {name} dataset [Done]. #samples: {len(candidate_dataset)}')

    # def __getitem__(self, index):
    #     sample = PIL.Image.open(os.path.join(self.root, "Images", self.path[index]))
    #     target = self.label[index]
    #     if self.transform is not None:
    #         sample = self.transform(sample)
    #     return sample, target, index

    # def __len__(self):
    #     return len(self.path)




def main():
    
    # loading data...
    dataset = Gaussian_dataset()
    # batch_size = 128
    # dataloader_new = torch.utils.data.DataLoader(dataset,
    #                                           batch_size=min(len(dataset), batch_size),
    #                                           shuffle=True,
    #                                           num_workers=1,
    #                                           drop_last=True)


    # for i, (img, target, idx) in enumerate(dataloader_new):
    #     import pdb
    #     pdb.set_trace()
    #     # img = img.unsqueeze(0).cuda(non_blocking=True)
    #     # target = target.cuda(non_blocking=True)
            

if __name__ == '__main__':
    main()