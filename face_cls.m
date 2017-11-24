% This is just a demo to load the .obj file
input_file_name = '../3D_face_dataset/Meshes/1010000319.obj';
[ node_num, face_num, normal_num, order_max ] = obj_size (input_file_name);

[ node_xyz, face_order, face_node ] = ...
    obj_read (input_file_name, node_num, face_num, normal_num, order_max);
% [vertices, faces] = obj__read(input_file_name);

