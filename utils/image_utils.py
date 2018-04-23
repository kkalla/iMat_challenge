# -*- coding: utf-8 -*-
"""
utils for image preprocessing

Created on Mon Apr 23 16:47:33 2018

@author: user
"""
import os

import numpy as np

from PIL import Image
from tqdm import tqdm

def reshape_and_save(image_dir,save_dir,file_name):
    """Reshaping image and save numpy arrays
    
    Save numpy array of shape ({'image_id':image_ids,'image_label':image_labels,
               'features':image_datasets})
    
    Arguments:
        image_dir: str, directory path where images are in
        save_dir: str, directory path to save numpy array
        file_name: str, file name of saved numpy array
    """
    dir_list = os.listdir(path=image_dir)
    print("Detect ",len(dir_list)," files.")
    print("Reshaping images...")
    
    image_datasets = []
    image_ids = []
    image_labels = []
    with tqdm(total=len(dir_list)) as t:
        for i in range(len(dir_list)):
            #Caching id and label of image
            _image_id = dir_list[i].split('.')[0].split('_')[0]
            _image_label = dir_list[i].split('.')[0].split('_')[1]
            #Reshaping
            image = Image.open(os.path.join(image_dir,dir_list[i]))
            image_resized = image.resize((800,800))
            image_list = list(image_resized.getdata())
            image_array = np.array(image_list,dtype='uint8').reshape((800,800,3))
            #Appending
            image_datasets.append(image_array)
            image_ids.append(_image_id)
            image_labels.append(_image_label)
            if i%1000==0:
                print("yeah!, ",np.array(image_datasets).shape)
            t.update(1)
    print("Done reshaping!!")
    
    #Converting 
    image_datasets = np.array(image_datasets)
    image_ids = np.array(image_ids)
    image_labels = np.array(image_labels)
    
    #Saving
    print("Saving dataset...")
    dataset = {'image_id':image_ids,'image_label':image_labels,
               'features':image_datasets}
    dataset = np.array(dataset)
    np.save(dataset,os.path.join(save_dir,file_name))
    
    
            
    