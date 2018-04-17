# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 21:37:16 2018

@author: kkalla
"""

import os, multiprocessing
import urllib.request
import argparse

from PIL import Image
from io import BytesIO
from tqdm import tqdm

from data_utils import Data_loader

parser = argparse.ArgumentParser()
parser.add_argument("--select-set",help="one of train/test/valid",type=str)
parser.add_argument("--save_dir",help="directory path to save images",type=str)
args = parser.parse_args()

def download_images(id_url_list):
    """
    Download images in id_url_list and save it to save_dir
    
    Args
    ------
    id_url_list: list, element is (image_id,url)
    save_dir: string, directory name or directory path to save images 
    
    """
    
    
    save_dir = args.save_dir
    
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    (image_id, url) = id_url_list
    file_name = os.path.join(save_dir,"{}.jpg".format(image_id))
    
    if os.path.exists(file_name):
        #print("Image #"+image_id+" already exists")
        return
    
    try:
        response = urllib.request.urlopen(url,timeout=30)
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
    which_set = args.select_set
    if which_set not in ['train','test','valid']:
        print("Warning: check --select-set!!")
        return
    train_list = get_id_url_list(which_set)
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    with tqdm(total=len(train_list)) as t:
        for _ in pool.imap_unordered(
                download_images,train_list):
            t.update(1)
    
    
if __name__=="__main__":
    main()