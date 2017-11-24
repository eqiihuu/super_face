function [node_xyz, face_node ]= obj__read( input_file_name )

%*****************************************************************************80
%
%% OBJ_DISPLAY displays the faces of a shape defined by an OBJ file.
%
%  Usage:
%
%    obj_display ( 'file.obj' )
%
%  Licensing:
%
%    This code is distributed under the GNU LGPL license.
%
%  Modified:
%
%    27 September 2008
%
%  Author:
%
%    John Burkardt
%
%  Get sizes.
%
  [ node_num, face_num, normal_num, order_max ] = obj_size ( input_file_name );
%
%  Get the data.
%
  [ node_xyz, face_order, face_node ] = ...
    obj_read ( input_file_name, node_num, face_num, normal_num, order_max );
%
%  FACE_NODE may contain polygons of different orders.
%  To make the call to PATCH, we will assume all the polygons are the same order.
%  To do so, we'll simply "stutter" the first node in each face list.
%
  for face = 1 : face_num
    face_node(face_order(face)+1:order_max,face) = face_node(1,face);
  end
%
%  If any node index is still less than 1, set the whole face to 1's.
%  We're giving up on this presumably meaningless face, but we need to
%  do it in a way that keeps MATLAB happy!
%
  for face = 1 : face_num
    for i = 1 : order_max
      face_node(i,face) = max ( face_node(i,face), 1 );
    end
  end
%
%  Display the shape.
%  The TITLE function will interpret underscores in the title.
%  We need to unescape such escape sequences!
  return
end