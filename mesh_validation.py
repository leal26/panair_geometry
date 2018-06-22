import numpy as np
import mesh_script as mesh
import panairwrapper as pw
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

file_wing_upper = 'models/ellip_upper.stl'
file_wing_lower = 'models/ellip_lower.stl'
file_wing_line_front = 'models/ellip_line_front.stl'
file_wing_line_back = 'models/ellip_line_back.stl'
file_wing_upper_line = 'models/ellip_upper_line.stl'
file_wing_lower_line = 'models/ellip_lower_line.stl'


wing_mesh = mesh.mesh_wing([file_wing_upper, file_wing_lower,
                            file_wing_line_front, file_wing_line_back,
                            file_wing_upper_line, file_wing_lower_line],
                            columns = 31, rows = 33,
                            wake = 10, wake_points = 21,
                            wing = 'right', function = 'linear',
                            spacing_c = 'cos', spacing_r = 'cos')
                            
mesh.scatter_plot(wing_mesh[0], wing_mesh[1], wing_mesh[3])


run_panair = True
if run_panair :
    case = pw.PanairWrapper('ellip2')
    case.add_network('wing_upper', wing_mesh[0])
    case.add_network('wing_lower', wing_mesh[1])
    case.add_network('wing_tip', wing_mesh[2])
    case.add_network('wing_root', wing_mesh[3])
    case.add_network('wing_wake', wing_mesh[4], network_type=18)
    case.set_aero_state(mach=0.01, alpha=6, beta=0)
    
    case.run()
    
    output =  pw.filehandling.OutputFiles('./panair_files')
    output.generate_vtk('ellip')

