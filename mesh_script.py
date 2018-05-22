import numpy as np
from math import *
import stl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.interpolate as si

'''
Notes :

X axis is supposed to be going from the nose to the tail of the plane.
Right side of the plane is positive Y coordinates, while left is negative.

'''

def points_from_stl(filename) :
    '''
    Generates an array of points from an .stl file
    Input : file name, as a string
    Output : array of shape (n, 3), n being the number of points and 3 being
    the x, y, z coordinates in that order.
    '''
    mesh = stl.mesh.Mesh.from_file(filename)
    list_points = []
    for p in mesh.v0 :
        list_points.append(p)
    for p in mesh.v1 :
        list_points.append(p)
    for p in mesh.v2 :
        list_points.append(p)
    return np.unique(list_points, axis=0)
    
    
def mesh_wing(files, columns=10, rows=10, wake=200, wake_points=10, 
              function='thin_plate', wing='right') :
              
    '''
    Generates a mesh from a wing, to be used as a Panair network.
    Includes a wake.
    Requires the wing to be split into several files to define boundaries.
    
    Inputs :
        files is a list of several filenames :
    files 1-2 are the upper and lower surfaces of the wing. They will be 
    used to interpolate most of the points
    files 3-4 are lines defining the leading and trailing edge respectively,
    they are needed for the boundaries of the wing.
    files 5-6 are lines defining the junction of the wing with the fuselage,
    for the upper and lower parts.
        columns is the number of points to be created along the span
        rows is the number of points to be created between the edges.
    upper and lower parts will each have that number of points.
        wake is the coordinate at which the wake should stop.
        wake points is the number of points should be created between the
    trailing edge and the end of the wake.
        function is the type of function to be used in the Rbf interpolation
    process. Default is 'thin_plate' as it seemed to work well for wings.
        wing should be 'right' or 'left' to define what wing is being meshed.
    Default is right. Please note that the mirror_mesh function is an easy way
    to generate the other wing.
    
    Output : an array of points of shape (4, c, n, 3), with 4 being the 
    different networks : upper side, lower side, wingtip and wake, c being 
    the number of columns, n the number of points in each column, and 3 being
    the x, y, z coordinates in that order.
    '''

    points_upper = points_from_stl(files[0])
    points_lower = points_from_stl(files[1])
    points_line_front = points_from_stl(files[2])
    points_line_back = points_from_stl(files[3])
    points_upper_line = points_from_stl(files[4])
    points_lower_line = points_from_stl(files[5])
    
    rbf_upper = si.Rbf(points_upper[:,0], points_upper[:,1], 
                       points_upper[:,2], function=function)
    rbf_lower = si.Rbf(points_lower[:,0], points_lower[:,1], 
                       points_lower[:,2], function=function)
    
    rbf_line_front_x = si.Rbf(points_line_front[:,1], points_line_front[:,0], 
                            function='linear')
    rbf_line_front_z = si.Rbf(points_line_front[:,1], points_line_front[:,2], 
                            function='linear')
    rbf_line_back_x = si.Rbf(points_line_back[:,1], points_line_back[:,0], 
                           function='linear')
    rbf_line_back_z = si.Rbf(points_line_back[:,1], points_line_back[:,2], 
                           function='linear')
    
    rbf_upper_line_y = si.Rbf(points_upper_line[:,0], points_upper_line[:,1], 
                              function='linear')
    rbf_upper_line_z = si.Rbf(points_upper_line[:,0], points_upper_line[:,2], 
                              function='linear')
    rbf_lower_line_y = si.Rbf(points_lower_line[:,0], points_lower_line[:,1], 
                              function='linear')
    rbf_lower_line_z = si.Rbf(points_lower_line[:,0], points_lower_line[:,2], 
                              function='linear')
    
    mesh_upper = []
    mesh_lower = []
    mesh_wake = []
    
    points_x = np.linspace(max(points_upper_line[:,0]), 
                           min(points_upper_line[:,0]), rows)
    point_A = [points_x[0], 
               rbf_upper_line_y(points_x[0]), rbf_upper_line_z(points_x[0])]
    col = []
    for x in points_x :
        col.append([x, rbf_upper_line_y(x), rbf_upper_line_z(x)])
    mesh_upper.append(col)
    
    points_x = np.linspace(min(points_lower_line[:,0]), 
                           max(points_lower_line[:,0]), rows)
    col=[]
    for x in points_x :
        col.append([x, rbf_lower_line_y(x), rbf_lower_line_z(x)])
    mesh_lower.append(col)
    
    points_x_wake = np.linspace(point_A[0], wake, wake_points)
    col = []
    for x in points_x_wake :
        col.append([x, point_A[1], point_A[2]])
    mesh_wake.append(col)
    
    
    if wing == 'right' :
        line_limit = max(max(np.array(mesh_upper[0])[:,1]), max(np.array(mesh_lower[0])[:,1]))
        wing_tip = max(points_upper[:,1])
    elif wing == 'left' :
        line_limit = min(min(np.array(mesh_upper[0])[:,1]), min(np.array(mesh_lower[0])[:,1]))
        wing_tip = min(points_upper[:,1])
    else :
        exit('Error : wing parameter must be "right" or "left" only')
        
    columns_y = np.linspace(line_limit, wing_tip, columns)[1:]
    
    for y in columns_y :
        
        points_x = np.linspace(rbf_line_back_x(y), rbf_line_front_x(y), rows)
        point_A = [points_x[0], y, rbf_line_back_z(y)]
        col = []
        col.append([points_x[0], y, rbf_line_back_z(y)])
        for x in points_x[1:-1] :
            col.append([x, y, rbf_upper(x, y)])
        col.append([points_x[-1], y, rbf_line_front_z(y)])
        mesh_upper.append(col)
            
        points_x = np.linspace(rbf_line_front_x(y), rbf_line_back_x(y), rows)
        col = []
        col.append([points_x[0], y, rbf_line_front_z(y)])
        for x in points_x[1:-1] :
            col.append([x, y, rbf_lower(x,y)])
        col.append([points_x[-1], y, rbf_line_back_z(y)])
        mesh_lower.append(col)
            
        points_x_wake = np.linspace(point_A[0], wake, wake_points)
        col = []
        for x in points_x_wake :
            col.append([x, point_A[1], point_A[2]])
        mesh_wake.append(col)
            
        mesh_tip = [mesh_upper[-1][::-1], mesh_lower[-1]]
        
    return [np.array(mesh_upper), np.array(mesh_lower),np.array(mesh_tip), np.array(mesh_wake)]


def mesh_part(files, axis=[0,1,2], columns=10, rows=10, function='cubic') :
    '''
    Generates a mesh for any part using a similiar process as the wing.
    Currently does not fit the panair expectations, work in progress.
    
    Inputs :
        files is a list of several filenames :
    files 1-2 are the 2 sides (upper/lower, left/right etc.) of the geometry.
    They will be used to interpolate the points
    files 3-4 are lines defining the edges (front/back, left/right etc.),
    they are needed for the boundaries of the wing.
        axis is a 3-element list defining the order of the axis : the first one
    is the axis of the columns, the second one is the axis of the rows.
        columns is the number of points to be created along the column axis
        rows is the number of points to be created between in every column.
    Each side of the geometry will have this number of points.
        function is the type of function to be used in the Rbf interpolation
    process. Default is 'cubic' but the right one to use really depends on
    the geometry.
    
    Output : an array of points of shape (c, n, 3), with c being the number 
    of columns, n the number of points in each column, and 3 being
    the x, y, z coordinates in that order.
    '''
    if len(axis) != 3 :
        exit('Number of axis incorrect for generation of mesh, must be 3')
    points_side1 = points_from_stl(files[0])
    points_side2 = points_from_stl(files[1])
    points_line1 = points_from_stl(files[2])
    points_line2 = points_from_stl(files[3])
    
    rbf_side1 = si.Rbf(points_side1[:,axis[0]], points_side1[:,axis[1]], 
                       points_side1[:,axis[2]], function=function, smooth=-100)
    rbf_side2 = si.Rbf(points_side2[:,axis[0]], points_side2[:,axis[1]], 
                       points_side2[:,axis[2]], function=function, smooth=-100)
    
    rbf_line1 = si.Rbf(points_line1[:,axis[0]], points_line1[:,axis[1]], 
                       function='linear')
    rbf_line2 = si.Rbf(points_line2[:,axis[0]], points_line2[:,axis[1]], 
                       function='linear')
    
    mesh_all = []
    
    points_columns = np.linspace(min(points_line1[:,axis[0]]), 
                                 max(points_line1[:,axis[0]]), columns)
    for c in points_columns :
        mesh_c = []
        line_points = [rbf_line1(c), rbf_line2(c)]
        points_rows = np.linspace(min(line_points), max(line_points), rows)
        for r in points_rows :
            mesh_c.append([c, r, rbf_side1(c, r)])
            
        points_rows = np.linspace(max(line_points), min(line_points), rows)
        for r in points_rows :
            mesh_c.append([c, r, rbf_side2(c, r)])
        
        mesh_all.append(mesh_c)
        
    return np.array(mesh_all)
    
def remove_double_points(mesh) :
    new_m = []
    for c in mesh :
        new_c = []
        for i in range(len(c)) :
            if len(new_c) == 0 :
                new_c.append(c[i])
            elif not np.array_equal(c[i], new_c[-1]) :
                new_c.append(c[i])
        new_m.append(new_c)
    return np.array(new_m)
    
def mirror_mesh(mesh, axis=1) :
    '''
    Mirrors a mesh along one axis. (Inverts each coordinate in that axis)
    Input is a mesh, and the axis (default is 1, for y axis). (0=x, 1=y, 2=z)
    Output is the mirrored mesh with the same shape as the original one.
    '''
    new_mesh = np.array(mesh)
    new_mesh[:,:,axis] *= -1
    return new_mesh
    
def scatter_plot(*meshes) :
    '''
    Display one or several meshes in a 3d plot.
    Bounds are scaled to fit entire model without deformation.
    '''
    bounds = []
    fig = plt.figure()
    ax3d = fig.add_subplot(111, projection='3d')
    
    for m in meshes :
        ax3d.scatter(m[:,:,0], m[:,:,1], m[:,:,2], marker='.')
        bounds_temp = bounds
        bounds_temp.append(min(m.flatten()))
        bounds_temp.append(max(m.flatten()))
        bounds = [min(bounds_temp), max(bounds_temp)]
        
    ax3d.set_aspect('equal','box')
    ax3d.set_xlabel('X')
    ax3d.set_xlim(bounds[0], bounds[1])
    ax3d.set_ylabel('Y')
    ax3d.set_ylim(bounds[0], bounds[1])
    ax3d.set_zlabel('Z')
    ax3d.set_zlim(bounds[0], bounds[1])
    
    plt.show()

