# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 16:13:59 2018

@author: kkalla
"""

import os, subprocess

class Data_loader():
    
    def __init__(self):
        self.url_dict = {'sample_submission':
        'https://www.kaggle.com/c/8220/download/sample_submission_randomlabel.csv',
        'train_json':'https://www.kaggle.com/c/8220/download/train.json',
        'test_json':'https://www.kaggle.com/c/8220/download/test.json',
        'valid_json':'https://www.kaggle.com/c/8220/download/validation.json'}
    
    def load_datasets(self,data_dir='data/'):
        for key in self.url_dict.keys():
            file_name = self.url_dict[key].split('/')[-1]
            file_dir = data_dir+file_name
            if not os.path.exists(file_dir):
                self._download_files(data_dir,file_name)

    def _download_files(self,data_dir,file_name):
        print(file_name + " is downloaded now...")
        subprocess.run(['kaggle','competitions','download','-c',
                        'imaterialist-challenge-furniture-2018','-p',data_dir,
                        '-f',file_name])
        print("Downloading is done!!")
        
    
        