import torch
import pickle
import numpy as np
from PIL import Image
import pickle
from datasets.BaseDataset import BaseDataset


class CXP(BaseDataset):
    def __init__(self, dataframe, path_to_pickles, sens_name, sens_classes, transform):
        super(CXP, self).__init__(dataframe, path_to_pickles, sens_name, sens_classes, transform)

        with open(path_to_pickles, 'rb') as f: 
            self.dataframe = pickle.load(f)
        self.transform = transform
        self.A = self.set_A(sens_name)
        self.Y = self.dataframe['binaryLabel']
        self.AY_proportion = self.get_AY_proportions()

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, idx):
        item = self.dataframe.iloc[idx % len(self)]

        img = Image.fromarray(item['image']).convert('RGB')
        img = self.transform(img)

        label = torch.Tensor([item['binaryLabel']]).float()
        #print(label, item['binaryLabel'])
        sensitive = self.get_sensitive(self.sens_name, self.sens_classes, item)
                
        return img, label, sensitive, idx
