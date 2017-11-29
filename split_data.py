# This is a script to split the dataset according to its label

from load_data import *
import shutil

label_path = './data/TDFN-Export (2014-08-20).txt'
data_dir = './data/2D_face'
male_dir = './data/2D_gender/male'
female_dir = './data/2D_gender/female'
labels, titles = read_label(label_path)
img_num = len(labels['ID'])
for i in range(img_num):
    old_path = data_dir+'/'+labels['ID'][i]+'.png'
    if labels['Sex'][i] == '1':
        new_path = female_dir+'/'+labels['ID'][i]+'.png'
    else:
        new_path = male_dir + '/' + labels['ID'][i] + '.png'
    shutil.copy(old_path, new_path)
