# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 17:33:54 2018

@author: user
"""

from data_utils import Data_loader

def test_data_loader():
    data_loader = Data_loader()
    train,test,valid,sample = data_loader.load_datasets(data_dir='../data')
    print(train.keys())
    
def main():
    test_data_loader()
    
if __name__ == '__main__':
    main()