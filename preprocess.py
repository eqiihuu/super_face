# This script is to pre-process the data

import os
from PIL import Image
from PIL import ImageFilter

img_size = 512
img_range = pow(2, 17)
data_dir = './data/Meshes'
new_data_dir = './data/2D_half'
files = os.listdir(data_dir)
id = 0
file_num = len(files)
for f_name in files[:]:
    print '%d/%d: %s'% (id, file_num, f_name)
    id += 1
    new_img_path = new_data_dir + '/' + f_name.split('.')[0] + '.png'
    if os.path.exists(new_img_path):
        continue
    f_path = data_dir + '/' + f_name
    f = open(f_path)
    line = f.readline()
    vertex_num = 0
    x_list = []
    y_list = []
    z_list = []
    # Skip the file-title part
    while line.split(' ')[0] != 'v':
        line = f.readline()
    # Read all vertexes
    while line != '':
        line = line.strip('\n').split(' ')
        if line[0] == '' or line[0] != 'v':
            break
        # print line
        vertex_num += 1
        x_list.append(float(line[1]))
        y_list.append(float(line[2]))
        z_list.append(float(line[3]))
        line = f.readline()
    f.close()
    x_min = min(x_list)
    x_max = max(x_list)
    y_min = min(y_list)
    y_max = max(y_list)
    z_min = min(z_list)
    z_max = max(z_list)
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    z_mean = sum(z_list)/len(z_list)

    img = Image.new('I', (img_size, img_size))
    for i in range(vertex_num):
        v_x = min(int(img_size * (x_max - x_list[i]) / x_range), img_size - 1)  # normalize and re-range
        v_y = min(int(img_size * (y_max - y_list[i]) / y_range), img_size - 1)
        v_z = int(img_range * (z_list[i] - z_min - z_range/2) / z_range)
        # print v_x, v_y, v_z

        img.putpixel((v_x, v_y), v_z*(v_z > 0))  # remove the back side
    img = img.filter(ImageFilter.MaxFilter(size=5))
    # img = img.filter(ImageFilter.MaxFilter(size=5))

    img.save(new_img_path, 'png')


# print 'Vertex Number: %d' % vertex_num
print 'X range: [%f, %f]' % (x_min, x_max)
print 'Y range: [%f, %f]' % (y_min, y_max)
print 'Z range: [%f, %f]' % (z_min, z_max)
