# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 16:13:59 2018

@author: kkalla
"""

import os, subprocess
import json

import pandas as pd
import numpy as np

from image_utils import reshape_and_save,convert_to_numpy

class Data_loader():
    
    def __init__(self,data_dir='data'):
        self.url_dict = {'sample_submission':
        'https://www.kaggle.com/c/8220/download/sample_submission_randomlabel.csv',
        'train_json':'https://www.kaggle.com/c/8220/download/train.json',
        'test_json':'https://www.kaggle.com/c/8220/download/test.json',
        'valid_json':'https://www.kaggle.com/c/8220/download/validation.json'}
        self.data_dir = data_dir
    
    def load_datasets(self,data_dir='data',which_set='all'):
        
        for key in self.url_dict.keys():
            file_name = self.url_dict[key].split('/')[-1]
            file_dir = os.path.join(data_dir,file_name)
            if not os.path.exists(file_dir):
                self._download_files(data_dir,file_name)
        
        train_set = json.load(open(os.path.join(data_dir,'train.json')))
        test_set = json.load(open(os.path.join(data_dir,'test.json')))
        valid_set = json.load(open(os.path.join(data_dir,'validation.json')))
        sample_submission_set = pd.read_csv(
                os.path.join(data_dir,'sample_submission_randomlabel.csv'))

        if which_set == "all":
            return train_set,test_set,valid_set,sample_submission_set
        if which_set == 'train':
            return train_set
        if which_set == 'test':
            return test_set
        if which_set == 'valid':
            return valid_set
        
    def load_image_data(self,data_dir):
        """Load train and valid images
        
        """
        train_images_dir = os.path.join(data_dir,'train_images')
        valid_images_dir = os.path.join(data_dir,'valid_images')
        train_dataset_path = os.path.join(train_images_dir,'train_dataset.npy')
        if not os.path.exists(train_dataset_path):
            print("There is no train_dataset.npy")
            if not len(os.listdir(train_images_dir)) == len(
                    os.listdir(os.path.join(data_dir,'train_images/resized'))):
                print("Execute reshaping and save")
                reshape_and_save(train_images_dir)
            print("Convert to numpy array...")
            convert_to_numpy(os.path.join(data_dir,'train_images/resized'),
                             train_dataset_path)
            print("Loading train dataset")
            train_dataset = np.load(train_dataset_path)
            print(train_dataset['features'].shape)
        else:
            print("Loading train dataset")
            train_dataset = np.load(train_dataset_path)
            print(train_dataset['features'].shape)
        
        return train_dataset
            
            
        
    def _download_files(self,save_dir,file_name):
        print(file_name + " is downloaded now...")
        subprocess.run(['kaggle','competitions','download','-c',
                        'imaterialist-challenge-furniture-2018','-p',save_dir,
                        '-f',file_name])
        print("Downloading is done!!")
        
    
        
