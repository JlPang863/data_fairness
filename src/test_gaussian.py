import numpy as np
import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt


class Gaussian_dataset(torch.utils.data.Dataset):

    def __init__(self, num_samples = 1000):
        scale = 2
        train_ratio = 0.8
        val_ratio = 0.1
        test_ratio = 1 - train_ratio
        all_data = np.random.randn(num_samples) * scale
        plt.hist(all_data)
        plt.show()




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
    batch_size = 128
    dataloader_new = torch.utils.data.DataLoader(dataset,
                                              batch_size=min(len(dataset), batch_size),
                                              shuffle=True,
                                              num_workers=1,
                                              drop_last=True)


    for i, (img, target, idx) in enumerate(dataloader_new):
        import pdb
        pdb.set_trace()
        # img = img.unsqueeze(0).cuda(non_blocking=True)
        # target = target.cuda(non_blocking=True)
            

if __name__ == '__main__':
    main()