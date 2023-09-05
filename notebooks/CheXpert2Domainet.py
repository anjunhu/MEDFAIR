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

DOMAIN = 'chexpert'
ATTRIBUTES = ['Atelectasis' ,'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
for TARGET_ATTRIBUTE in ['Cardiomegaly']:
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

        # unify the value of sensitive attributes
        sex = demo_data['Sex'].values
        sex[sex == 'Male'] = 'M'
        sex[sex == 'Female'] = 'F'
        demo_data['Sex'] = sex
        ta = demo_data[TARGET_ATTRIBUTE].values
        print('TARGET_ATTRIBUTE', 'SPLIT', 'ONES', 'ZEROS', 'NEGONES', 'NANS')
        print(TARGET_ATTRIBUTE, SPLIT, len(ta[ta==1]), len(ta[ta==0]), len(ta[ta==-1]), len(ta[ta!=ta]))
        print(len(ta[ta==1])/len(ta), len(ta[ta==0])/len(ta), len(ta[ta==-1])/len(ta), len(ta[ta!=ta])/len(ta))

        # Ignore Uncertain and Emtpy entires
        #demo_data = demo_data[~demo_data[TARGET_ATTRIBUTE].isnull()]
        #assert(len(ta[ta!=ta])==0)
        demo_data = demo_data[demo_data[TARGET_ATTRIBUTE]!=-1.0]
        ta = demo_data[TARGET_ATTRIBUTE].values
        assert(len(ta[ta==-1])==0)
        ta[ta != 1.0] = 0.0
        demo_data[TARGET_ATTRIBUTE] = ta

        # Null to negative
        ta = demo_data[TARGET_ATTRIBUTE].values.copy()
        ta[ta != ta] = '0.0'
        ta = ta.astype('int')
        demo_data[TARGET_ATTRIBUTE] = ta
        demo_data['binaryLabel'] = ta

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

        lines = []
        start = time.time()
        ta = TARGET_ATTRIBUTE.replace(' ', '')
        path = '/home/shared_space/data/'
        for i in tqdm(range(len(demo_data))):
            os.makedirs(os.path.join(f'domainnet_style_datasets_{ta}', DOMAIN, str(int(demo_data.iloc[i]['binaryLabel']))), exist_ok=True) 
            img = cv2.imread(path + demo_data.iloc[i]['Path'])
            # resize to the input size in advance to save time during training
            img = cv2.resize(img, (256, 256))
            filename = demo_data.iloc[i]['Path'].split("/")
            filename = os.path.join(DOMAIN, str(int(demo_data.iloc[i]['binaryLabel'])), '_'.join(filename[-3:]))
            lines.append(filename+' '+str(int(demo_data.iloc[i]['binaryLabel']))+' '+str(int(demo_data.iloc[i]['Age_multi'])))
            print(lines[-1])
            cv2.imwrite(os.path.join(f'domainnet_style_datasets_{ta}', filename), img)
        end = time.time()
        print('Time Elapsed', end-start)

        with open('/home/scat9241/repos/MEDFAIR/' + f'domainnet_style_datasets_{ta}/chexpert_list.txt', 'w') as f:
            f.write('\n'.join(lines))
