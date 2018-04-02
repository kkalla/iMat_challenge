# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 21:37:16 2018

@author: kkalla
"""

import os, multiprocessing
import urllib.request

from PIL import Image
from io import BytesIO
from tqdm import tqdm

from data_utils import Data_loader


def download_images(id_url_list,save_dir):
    """
    Download images in id_url_list and save it to save_dir
    
    Args
    ------
    id_url_list: list, element is (image_id,url)
    save_dir: string, directory name or directory path to save images 
    
    """
    
    (image_id, url) = id_url_list
    file_name = os.path.join(save_dir,"{}.jpg".format(image_id))
    
    if os.path.exists(file_name):
        print("Image #"+image_id+" already exists")
        return
    
    try:
        response = urllib.request.urlopen(url)
        image_data = response.read()
    except:
        print('Can not download image #'+image_id+' from '+url)
        return

    try:      
        pil_image = Image.open(BytesIO(image_data))
    except:
        print('Warning: Failed to parse image #{},{}'.format(image_id,url))
        return
    
    try:
        pil_image_rgb = pil_image.convert('RGB')
    except:
        print('Warning: Failed to convert to RGB image#'+image_id)
        return
    
    try:
        pil_image_rgb.save(file_name,format='JPEG',quality=90)
    except:
        print('Warning: Failed to save image #{}'.format(image_id))
        return
    
    
def get_id_url_list(which_set):
    data_loader = Data_loader()
    selected_set = data_loader.load_datasets(data_dir='../data',which_set=which_set)
    if which_set == 'train':
        ann = selected_set['annotations']
        id_label_list = {}
        for a in ann:
            id_label_list[a['image_id']] = a['label_id']    
    
    id_url_list = []
    images = selected_set['images']
    for item in images:
        url = item['url'][0]
        image_id = item['image_id']
        if which_set == 'train':
            image_id = "{}_{}".format(image_id, id_label_list[image_id])
        id_url_list.append((image_id,url))
    return id_url_list
    
def main():
    train_list = get_id_url_list('train')
    pool = multiprocessing.Pool(processes=3)
    with tqdm(total=len(train_list)) as t:
        for _ in pool.imap_unordered(
                download_images(train_list,save_dir='../data/train_images')):
            t.update(1)
    
    
if __name__=="__main__":
    main()