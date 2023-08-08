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

for TARGET_ATTRIBUTE in ['Atelectasis' ,'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion', 'Pneumothorax', 'Pneumonia']:
    for SPLIT in ['train', 'valid']:
        # read metadata
        path = '/home/shared_space/data/CheXpert-v1.0-small/'
        #demo_data = pd.read_csv(path + f'{SPLIT}.csv')
        train_df = pd.read_csv(path + f'train.csv')
        valid_df = pd.read_csv(path + f'valid.csv')
        demo_data = pd.concat([train_df, valid_df])

        # remove age/sex == null
        #demo_data = demo_data[~demo_data['Age'].isnull()]
        #demo_data = demo_data[~demo_data['Sex'].isnull()]
        #demo_data = demo_data[demo_data[TARGET_ATTRIBUTE]!=-1.0]

        # unify the value of sensitive attributes
        sex = demo_data['Sex'].values
        sex[sex == 'Male'] = 'M'
        sex[sex == 'Female'] = 'F'
        demo_data['Sex'] = sex
        ta = demo_data[TARGET_ATTRIBUTE].values
        print('TARGET_ATTRIBUTE', 'SPLIT', 'ONES', 'ZEROS', 'NEGONES', 'NANS')
        print(TARGET_ATTRIBUTE, SPLIT, len(ta[ta==1]), len(ta[ta==0]), len(ta[ta==-1]), len(ta[ta!=ta]))
        print(len(ta[ta==1])/len(ta), len(ta[ta==0])/len(ta), len(ta[ta==-1])/len(ta), len(ta[ta!=ta])/len(ta))
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

        ta = TARGET_ATTRIBUTE.replace(' ', '')

        train, valid = train_test_split(demo_data, test_size=0.25)

        print(ta, SPLIT, 'train', len(train), 'val', len(valid))
        with open('/home/scat9241/repos/MEDFAIR/' + f'pickled_datasets/CXP/train_{ta}.pkl', 'wb') as f:
            pickle.dump(train, f)

        with open('/home/scat9241/repos/MEDFAIR/' + f'pickled_datasets/CXP/valid_{ta}.pkl', 'wb') as f:
            pickle.dump(valid, f)

        '''
        df_male = demo_data[demo_data['Sex']=='M']
        df_female = demo_data[demo_data['Sex']=='F']
        df_over60 = demo_data[demo_data['Age_binary']==1]
        df_under60 = demo_data[demo_data['Age_binary']==0]

        print(ta, SPLIT, 'male', len(df_male), 'female', len(df_female))
        print(ta, SPLIT, 'over60', len(df_over60), 'under60', len(df_under60))

        with open('/home/scat9241/repos/MEDFAIR/' + f'pickled_datasets/CXP/{SPLIT}_male_{ta}.pkl', 'wb') as f:
            pickle.dump(df_male, f)
        with open('/home/scat9241/repos/MEDFAIR/' + f'pickled_datasets/CXP/{SPLIT}_female_{ta}.pkl', 'wb') as f:
            pickle.dump(df_female, f)
        with open('/home/scat9241/repos/MEDFAIR/' + f'pickled_datasets/CXP/{SPLIT}_over60_{ta}.pkl', 'wb') as f:
            pickle.dump(df_over60, f)
        with open('/home/scat9241/repos/MEDFAIR/' + f'pickled_datasets/CXP/{SPLIT}_under60_{ta}.pkl', 'wb') as f:
            pickle.dump(df_under60, f)
        '''
