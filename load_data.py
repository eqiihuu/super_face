# This is a script to define some tool functions for loading data
import os
import random
import numpy as np
from PIL import Image


# Load labels
def read_label(f_path):
    f = open(f_path)
    labels = {}
    title_str = f.readline()
    titles = read_title(title_str)
    title_num = len(titles)
    line = f.readline()
    while line != '':
        line_s = line.strip('\n').split(',')
        id = line_s[0].strip('"')
        labels[id] = {}
        for i in range(1, title_num-1):
            # print line_s[i].strip('"')
            labels[id][titles[i]]= line_s[i].strip('"')
        line = f.readline()
    return labels, titles


# Read the title part of label-file
def read_title(title_str):
    titles = title_str.split(',')
    for i in range(len(titles)):
        t = titles[i].strip('"')
        titles[i] = t
    return titles


def read_image(image_dir):
    images = os.listdir(image_dir)
    image_list = []
    id_list = []
    for image_name in images:
        image_id = image_name.split('.')[0]
        image_path = os.path.join(image_dir, image_name)
        image = Image.open(image_path, 'r')
        image_list.append(np.asarray(image))
        id_list.append(image_id)
    return image_list, id_list


def get_label_list(labels, id_list, title):
    label_list = []
    for id in id_list:
        label_list.append(labels[id][title])
    return label_list


def read_image_label(image_dir, label_path, title):
    '''
    Read the training/validation/testing images and corresponding labels
    Parameters:
        image_dir:
        label_path:
    Return:
        image_list
        label_list
    '''
    image_list, id_list = read_image(image_dir)
    labels, titles = read_label(label_path)
    label_list = get_label_list(labels, id_list, title)
    return id_list, image_list, label_list

if __name__ == '__main__':
    image_dir = './data/2D_face_256'
    label_path = './data/TDFN-Export (2014-08-20).txt'
    id_list, image_list, label_list = read_image_label(image_dir, label_path, 'Sex')
    print id_list[0]
    print image_list[0]
    print label_list[0]
