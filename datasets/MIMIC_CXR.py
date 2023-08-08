import torch
import pickle
import numpy as np
from PIL import Image
import pickle
from datasets.BaseDataset import BaseDataset


class MIMIC_CXR(BaseDataset):
    def __init__(self, dataframe, path_to_pickles, sens_name, sens_classes, transform):
        super(MIMIC_CXR, self).__init__(dataframe, path_to_pickles, sens_name, sens_classes, transform)

        with open(path_to_pickles, 'rb') as f:
            self.dataframe = pickle.load(f)
        self.transform = transform
        print(self.dataframe )
        self.A = self.set_A('binaryLabel') #(sens_name)
        self.Y = self.dataframe['binaryLabel']
        self.AY_proportion = self.get_AY_proportions()

    def __getitem__(self, idx):
        item = self.dataframe.iloc[idx % len(self)]

        img = Image.fromarray(item['image']).convert('RGB')
        img = self.transform(img)

        label = torch.Tensor([item['binaryLabel']]).float()
        #print(label, item['binaryLabel'])
        sensitive = torch.Tensor([0.]).float() #self.get_sensitive(self.sens_name, self.sens_classes, item)

        return img, label, sensitive, idx
