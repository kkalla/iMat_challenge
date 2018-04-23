# -*- coding: utf-8 -*-
"""
utils for image preprocessing

Created on Mon Apr 23 16:47:33 2018

@author: user
"""
import os,multiprocessing

import numpy as np

from PIL import Image
from tqdm import tqdm

def reshape_and_save(image_dir):
    """Reshaping image and save resized_image
    It saves images at 'image_dir/resized' directory
    
    Arguments:
        image_dir: str, directory path where images are in
        save_dir: str, directory path to save resized image
    """
    dir_list = os.listdir(path=image_dir)
    print("Detect ",len(dir_list)," files.")
    print("Reshaping images...")
    
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    image_list = [os.path.join(image_dir,file_name) for file_name in dir_list]
    with tqdm(total=len(image_list)) as t:
       for _ in pool.imap_unordered(reshape,image_list):
           t.update(1)
    print("Done reshaping!!")
    
def reshape(image_loc):
    if not os.path.isdir(image_loc):
        image_dir = os.path.split(image_loc)[0]
        file_name = os.path.split(image_loc)[1]
        image = Image.open(image_loc)
        image_resized = image.resize((800,800))
        save_dir = os.path.join(image_dir,'resized')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        if not os.path.exists(os.path.join(save_dir,file_name)):
            image_resized.save(os.path.join(save_dir,file_name),
                           format='JPEG',quality=100)
    
        
    
    
def convert_to_numpy(resized_image_dir,save_dir):
    """Convert images to numpy array and save as .npy file
    include images ids and labels
    """
    dir_list = os.listdir(path=resized_image_dir)
    image_data = []
    image_ids = []
    image_labels = []
    k = 0
    for i in dir_list:
        image_file = os.path.join(resized_image_dir,i)
        image = Image.open(image_file)
        _image_id = i.split('.')[0].split('_')[0]
        _image_label = i.split('.')[0].split('_')[1]
        _image_data = np.array(list(image.getdata())).reshape((800,800,3))
        
        image_data.append(_image_data)
        image_ids.append(_image_id)
        image_labels.append(_image_label)
        if k%1000==0:
            print("# ",k,"running....")
        k+=1
    
    image_dataset = {'image_id':image_ids,'image_label':image_labels,
                     'features':image_data}
    image_dataset = np.array(image_dataset)
    np.save(image_dataset,save_dir)
    
    
    
            
    
