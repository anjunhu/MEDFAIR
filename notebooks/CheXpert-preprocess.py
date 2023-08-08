import h5py
import pandas as pd
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
import pickle
import time
from tqdm import tqdm


for TARGET_ATTRIBUTE in ['Pneumonia']:
    for SPLIT in ['train', 'valid']:
        # read metadata
        path = '/home/shared_space/data/CheXpert-v1.0-small/'
        demo_data = pd.read_csv(path + f'{SPLIT}.csv')
        # remove age/sex == null
        demo_data = demo_data[~demo_data['Age'].isnull()]
        demo_data = demo_data[~demo_data['Sex'].isnull()]
        demo_data = demo_data[demo_data[TARGET_ATTRIBUTE]!=-1.0]

        # unify the value of sensitive attributes
        sex = demo_data['Sex'].values
        sex[sex == 'Male'] = 'M'
        sex[sex == 'Female'] = 'F'
        demo_data['Sex'] = sex
        ta = demo_data[TARGET_ATTRIBUTE].values
        ta[ta != 1.0] = 0.0
        demo_data[TARGET_ATTRIBUTE] = ta

        # split subjects to different age groups
        demo_data['Age_multi'] = demo_data['Age'].values.astype('int')
        demo_data['Age_multi'] = np.where(demo_data['Age_multi'].between(-1,19), 0, demo_data['Age_multi'])
        demo_data['Age_multi'] = np.where(demo_data['Age_multi'].between(20,39), 1, demo_data['Age_multi'])
        demo_data['Age_multi'] = np.where(demo_data['Age_multi'].between(40,59), 2, demo_data['Age_multi'])
        demo_data['Age_multi'] = np.where(demo_data['Age_multi'].between(60,79), 3, demo_data['Age_multi'])
        demo_data['Age_multi'] = np.where(demo_data['Age_multi']>=80, 4, demo_data['Age_multi'])

        demo_data['Age_binary'] = demo_data['Age'].values.astype('int')
        demo_data['Age_binary'] = np.where(demo_data['Age_binary'].between(-1, 60), 0, demo_data['Age_binary'])
        demo_data['Age_binary'] = np.where(demo_data['Age_binary']>= 60, 1, demo_data['Age_binary'])

        labels = demo_data[TARGET_ATTRIBUTE].values.copy()
        labels[labels != labels] = '0.0'
        labels = labels.astype('int')
        demo_data['binaryLabel'] = labels

        images = []
        start = time.time()
        path = '/home/shared_space/data/'
        for i in tqdm(range(len(demo_data))):
            img = cv2.imread(path + demo_data.iloc[i]['Path'])
            # resize to the input size in advance to save time during training
            img = cv2.resize(img, (256, 256))
            images.append(img)
        demo_data['image'] = images
    
        end = time.time()
        print(end-start)
        ta = TARGET_ATTRIBUTE.replace(' ', '_')
        with open('/home/scat9241/repos/MEDFAIR/' + f'pickled_datasets/CXP/{SPLIT}_{ta}.pkl', 'wb') as f:
            pickle.dump(demo_data, f)

        with open('/home/scat9241/repos/MEDFAIR/' + f'pickled_datasets/CXP/{SPLIT}_{ta}.pkl', 'rb') as f:
            data = pickle.load(f)

        print(data)
