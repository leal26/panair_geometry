import numpy as np
import mesh_script as mesh
import panairwrapper as pw
import matplotlib.pyplot as plt


file_wing_upper = 'models/25d_wing_upper_ref.stl'
file_wing_lower = 'models/25d_wing_lower_ref.stl'
file_wing_line_front = 'models/25d_wing_line_front.stl'
file_wing_line_back = 'models/25d_wing_line_back.stl'
file_wing_upper_line = 'models/25d_wing_upper_line.stl'
file_wing_lower_line = 'models/25d_wing_lower_line.stl'


wing_mesh = mesh.mesh_wing([file_wing_upper, file_wing_lower,
                            file_wing_line_front, file_wing_line_back,
                            file_wing_upper_line, file_wing_lower_line],
                            columns = 21, rows = 31,
                            wake = 20, wake_points = 11,
                            wing = 'right', scale=0.1,
                            function = 'linear')


mesh.scatter_plot(wing_mesh[0], wing_mesh[1], wing_mesh[2], wing_mesh[3], wing_mesh[4])


run_panair = True
if run_panair :
    case = pw.PanairWrapper('25D')
    case.add_network('wing_upper', wing_mesh[0])
    case.add_network('wing_lower', wing_mesh[1])
    case.add_network('wing_tip', wing_mesh[2])
    case.add_network('wing_root', wing_mesh[3])
    case.add_network('wing_wake', wing_mesh[4], network_type=18)
    case.set_aero_state(mach=0.01, alpha=0, beta=0)
    
    case.run()
