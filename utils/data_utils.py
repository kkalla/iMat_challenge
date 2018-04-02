# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 16:13:59 2018

@author: kkalla
"""

import os, subprocess
import json

import pandas as pd

class Data_loader():
    
    def __init__(self):
        self.url_dict = {'sample_submission':
        'https://www.kaggle.com/c/8220/download/sample_submission_randomlabel.csv',
        'train_json':'https://www.kaggle.com/c/8220/download/train.json',
        'test_json':'https://www.kaggle.com/c/8220/download/test.json',
        'valid_json':'https://www.kaggle.com/c/8220/download/validation.json'}
            
    
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
            
        
    def _download_files(self,save_dir,file_name):
        print(file_name + " is downloaded now...")
        subprocess.run(['kaggle','competitions','download','-c',
                        'imaterialist-challenge-furniture-2018','-p',save_dir,
                        '-f',file_name])
        print("Downloading is done!!")
        
    
        