# This script is to get the range of x, y, z of all images

import os

data_dir = './data/Meshes'
files = os.listdir(data_dir)
id = 0
file_num = len(files)
for f_name in files[:]:
    print '%d/%d: %s'% (id, file_num, f_name)
    id += 1
    f_path = data_dir + '/' + f_name
    f = open(f_path)
    line = f.readline()
    vertex_num = 0
    x_list = []
    y_list = []
    z_list = []
    x_min = 1000
    x_max = -1000
    y_min = 1000
    y_max = -1000
    z_min = 1000
    z_max = -1000
    # Skip the file-title part
    while line.split(' ')[0] != 'v':
        line = f.readline()
    # Read all vertexes
    while line != '':
        line = line.strip('\n').split(' ')
        if line[0] == '' or line[0] != 'v':
            break
        # print line
        # vertex_num += 1
        x_list.append(float(line[1]))
        y_list.append(float(line[2]))
        z_list.append(float(line[3]))
        line = f.readline()
    f.close()
    x_min = min(x_min, min(x_list))
    x_max = max(x_max, max(x_list))
    y_min = min(y_min, min(y_list))
    y_max = max(y_max, max(y_list))
    z_min = min(z_min, min(z_list))
    z_max = max(z_max, max(z_list))


# print 'Vertex Number: %d' % vertex_num
print 'X range: [%f, %f]' % (x_min, x_max)
print 'Y range: [%f, %f]' % (y_min, y_max)
print 'Z range: [%f, %f]' % (z_min, z_max)
