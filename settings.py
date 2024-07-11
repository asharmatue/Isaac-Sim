from shapes import *

##### Research Paper Parameters #####
r_avoid = 0.4 # 1.5 in Paper 				# Hardcode avoid radius of robots, used for calculating l_cell
r_sense = 0.7 # 2.5 in Paper				# Hardcode sense radius of robots, used for calculating neighboring_i() and neighbouring_cells()
r_check = 1.2 # equal to r_avoid in Paper	# Hardcode check radius for walls

entering_weight = 0.9						# Hardcode velocity commands weigths
exploration_weight = 4.3 					
interaction_weight = 0.35 					

c_1 = 1.6       # 1.6 in Paper				# Hardcode parameters for calculate_v_rho0_i()
alpha = 0.8     # 0.8 in Paper				# |
k_1 = 10		# 10 in Paper				# Hardcode parameter shape_entering_velocity()
k_3 = 25    	# 20,25,30 in Paper			# Hardcode parameter interaction_velocity()

forward_gain = 0.02							# Hardcode parameters for differential drive
angle_gain = 0.8							# |

# Modifer
remove_redundant_obstacle_positions = True	# Remove other robots' positions in find_collision_points()
######################################


##### Visualisation Parameters #####
show_vel_spheres = False						# Show Velocity command contributions visually?
show_robot_obstacle_positions = False			# Show considerd Collision Points visually?
show_grid_vis = True							# Show Greyscale grid visually?
show_old_grid_vis = False						# Show grid lines visually?
#####################################

##### Print\Log Parameters #####
# main.py
show_log_get_robot_target_rho = False
show_log_find_collision_points = False
show_interaction_velocity = False
show_log_in_shape_boundary = False
show_log_neighbouring_cells = False
show_log_shape_exploration_velocity = False

show_log_velocity_commands = True
show_log_send_robot_actions = True

# grid.py
show_log_grid = False
################################

##### Robot Parameters #####
num_robots = 3 					# Number of robots
lineMode = True					# True: Robots spawn in a line; False: Robots spawn in a grid
typeRobot = 1       			# 1 -> Jetbot, 2 -> Kaya
lidarsDrawLines = False#True	# Show lidar lines?
lidarsDrawPoints = False		# Show lidar points of surface?

r_body_size = 0.16 				# Robot dimensions 0.160 x 0.135 x 0.260 [X,Y,Z]
r_body = 0.09					# Robot is -0.09 away from center in X, so should take that as widest	
############################


##### Environment Parameters #####
input_shape = shape_array_floorplan		# Navigation shape that robots should explore.
# h = np.ceil((n_cell ** 0.5)/2)		# Calculated number of iterations for greyscale()
h = 4 	 								# Hardcode Number of iterations for greyscale()

# actual_environment_x_min = -1.4		# Hardcode actual enviroment size (of biggest room) in meters
# actual_environment_x_max = 1.4
# actual_environment_y_min = -0.4
# actual_environment_y_max = 0.8

show_walls = True
show_door = False

show_test_wall = False
show_victim_cube = False

a_e_v_amount = 3						# Hardcode set enviroment size in meters to test shape_entering_velocity()
actual_environment_x_min = -a_e_v_amount
actual_environment_x_max = a_e_v_amount
actual_environment_y_min = -a_e_v_amount
actual_environment_y_max = a_e_v_amount
actual_environment_size_x = actual_environment_x_max - actual_environment_x_min
actual_environment_size_y = actual_environment_y_max - actual_environment_y_min
##################################

##### Situation Parameters #####

SITUATION_NAME = "three_rooms_detail"
SITUATIONS_PATH = "situations"

################################

##### Timeline Parameters #####

SAVE_TIMELINE = True
SEND_TIMELINE = False

TIMELINE_PATH = "timelines"

LIMIT_TIMELINE_RATE = True
TIMELINE_RATE = 1 / 8 # Minimum time between saves in seconds.

################################

##### Export Parameters #####

EXPORT_DATA = True
EXPORT_FILE_ADDRESS = "export/three_rooms_detail_data"

################################

##### Communicator Parameters #####

PORT_NUMBER = 5646
SERVER_IP = "127.0.0.1"
CREATE_CONNECTION = True

################################

##### Utility Parameters #####

DEBUG_LOG = False
LOG_CONTEXT_SPACE = 29
MEASURE_PERFORMANCE = True

###############################