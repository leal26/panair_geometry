import numpy as np
import stl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

'''
Script to read a panair input file and make it into and display it or make it
into an .stl file.
'''

def count_numbers(string) :
    count = 0
    for c in string :
        if c.isdigit() :
            count += 1
    return count

    
def read_input_file(filename) :
    points = []
    file = open(filename)
    line = file.readline()
    while line[:4] != '$end' :
        length = len(line)
        if length >= 31 :
            numbers = count_numbers(line)
            if float(numbers)/length >= 0.5 :
                point1 = [float(line[0:10]), 
                          float(line[10:20]), 
                          float(line[20:30])]
                points.append(point1)
                if length >= 61 :
                    point2 = [float(line[30:40]), 
                              float(line[40:50]), 
                              float(line[50:60])]
                    points.append(point2)
        line = file.readline()
    return np.array(points)

    
def scatter_points(points, title=None) :
    fig = plt.figure()
    ax3d = fig.add_subplot(111, projection='3d')
    
    ax3d.scatter(points[:,0], points[:,1], points[:,2], marker='.')
    
    space = max([max(points[:,0])-min(points[:,0]),
                 max(points[:,1])-min(points[:,1]),
                 max(points[:,2])-min(points[:,2])])
    x_bounds = [(max(points[:,0])+min(points[:,0]))/2-space/2,
                (max(points[:,0])+min(points[:,0]))/2+space/2]
    y_bounds = [(max(points[:,1])+min(points[:,1]))/2-space/2,
                (max(points[:,1])+min(points[:,1]))/2+space/2]
    z_bounds = [(max(points[:,2])+min(points[:,2]))/2-space/2,
                (max(points[:,2])+min(points[:,2]))/2+space/2]
        
    if title is not None :
        ax3d.set_title(title)
    ax3d.set_aspect('equal','box')
    ax3d.set_xlabel('X')
    ax3d.set_xlim(x_bounds[0], x_bounds[1])
    ax3d.set_ylabel('Y')
    ax3d.set_ylim(y_bounds[0], y_bounds[1])
    ax3d.set_zlabel('Z')
    ax3d.set_zlim(z_bounds[0], z_bounds[1])
    
    plt.show()
    
    
def make_stl(points, filename) :
    n = len(points)
    mesh = stl.mesh.Mesh(np.zeros((n-2,3,3), dtype=stl.mesh.Mesh.dtype))
    for i in range(n)[:-2] :
        mesh.vectors[i] = [points[i], points[i+1], points[i+2]]
    mesh.save(filename)
    
    
    
if __name__ == '__main__' :
    points = read_input_file('panair_files/25D.inp')
    scatter_points(points, '25D')
    make_stl(points, '25D.stl')
    