import sys
import os
import pandas as pd 
import numpy as np
from utils import hdf5_reader
from skimage import measure



def csv_maker(input_path,save_path,label_list):
    csv_info = []
    entry = os.scandir(input_path)
    len_ = len(os.listdir(input_path))
    count = 0 
    
    for item in entry:
        csv_item = []
        csv_item.append(item.path)
        tag_array = np.zeros((len(label_list) + 1,),dtype=np.uint8)
        label = hdf5_reader(item.path,'label')
        # print(np.unique(label).astype(np.uint8))
        tag_array[np.unique(label).astype(np.uint8)] = 1
        csv_item.extend(list(tag_array[1:]))
        # print(item.path)
        # print(list(tag_array[1:]))
        csv_info.append(csv_item)
        count += 1
        sys.stdout.write('\r Current: %d / %d'%(count,len_))
    print('\n')
    col = ['path'] + label_list
    csv_file = pd.DataFrame(columns=col, data=csv_info)
    csv_file.to_csv(save_path, index=False)


def area_compute(input_path,save_path,label_list):
    csv_info = []
    entry = os.scandir(input_path)
    len_ = len(os.listdir(input_path))
    count = 0 
    
    for item in entry:
        csv_item = []
        csv_item.append(item.path)
        area_array = list(np.zeros((len(label_list),),dtype=np.uint8))
        label = hdf5_reader(item.path,'label')
        for i in range(len(label_list)):
            roi = (label==i+1).astype(np.uint8)
            roi = measure.label(roi)
            area = []
            for j in range(1,np.amax(roi) + 1):
                area.append(np.sum(roi == j))
            area_array[i] = area
        
        csv_item.extend(area_array)

        csv_info.append(csv_item)
        count += 1
        sys.stdout.write('\r Current: %d / %d'%(count,len_))
    print('\n')
    col = ['path'] + label_list
    csv_file = pd.DataFrame(columns=col, data=csv_info)
    csv_file.to_csv(save_path, index=False)


if __name__ == "__main__":
    lab_path = 'data_path'
    csv_path = './area.csv'
    # csv_path = './annotation.csv'
    label_list = [1]
    # csv_maker(lab_path,csv_path,label_list)
    area_compute(lab_path,csv_path,label_list)
