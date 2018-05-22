import numpy as np
import mesh_script as mesh
import panairwrapper as pw


file_wing_upper = 'models/25d_wing_upper_ref.stl'
file_wing_lower = 'models/25d_wing_lower_ref.stl'
file_wing_line_front = 'models/25d_wing_line_front.stl'
file_wing_line_back = 'models/25d_wing_line_back.stl'
file_wing_upper_line = 'models/25d_wing_upper_line.stl'
file_wing_lower_line = 'models/25d_wing_lower_line.stl'

file_fuselage_upper = 'models/25d_fuselage_upper_ref.stl'
file_fuselage_lower = 'models/25d_fuselage_lower_ref.stl'
file_fuselage_line_left = 'models/25d_fuselage_line_left.stl'
file_fuselage_line_right = 'models/25d_fuselage_line_right.stl'

wing_mesh = mesh.mesh_wing([file_wing_upper, file_wing_lower,
                            file_wing_line_front, file_wing_line_back,
                            file_wing_upper_line, file_wing_lower_line],
                            columns = 10, rows = 21,
                            wake = 200, wake_points = 11,
                            function = 'thin_plate',
                            wing = 'right')
                            
                       
fuselage_mesh = mesh.mesh_part([file_fuselage_upper, file_fuselage_lower,
                                file_fuselage_line_left, file_fuselage_line_right],
                                columns = 50, rows = 10,
                                function = 'cubic')
                                

mesh.scatter_plot(wing_mesh[0], wing_mesh[1], wing_mesh[2], wing_mesh[3], fuselage_mesh)


run_panair = False
if run_panair :
    case = pw.PanairWrapper('25D')
    case.add_network('wing_upper', wing_mesh[0])
    case.add_network('wing_lower', wing_mesh[1])
    case.add_network('wing_tip', wing_mesh[2])
    case.add_network('wing_wake', wing_mesh[3])
    case.set_aero_state(mach=1.6, alpha=0, beta=0)
    
    case.run()