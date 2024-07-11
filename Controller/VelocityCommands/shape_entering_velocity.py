import numpy as np


from grid import grey_grid, normalized_x_steps, normalized_y_steps
from settings import num_robots, c_1, alpha, k_1
from settings import actual_environment_x_min, actual_environment_y_min, show_log_get_robot_target_rho
from util import log
from Perception.neighbors import neighboring_i

# Inital v_rho0_i set to 0
robs_initial_v_rho0_i = [[0,0,0] for _ in range(num_robots)]

def initialize_v_rho0_i(robots):
	# Changing inital v_rho0_i from [0,0,0] to become get_robot_vel() initially. 
    # Afterwards these values are overwritten in calculate_v_rho0_i() as intented
    for robot_index in range(num_robots):
        robs_initial_v_rho0_i[robot_index] = robots[robot_index].vel

def calculate_v_rho0_i(robots, robot_index):
	"""
	Calculate the local interpretation of the moving velocity of the entire shape.
	Returns:
	- v_rho_i: The local interpretation of the moving velocity of the entire shape for robot i.
	"""

	N = neighboring_i(robots, robot_index)

	if len(N) == 0:
		log("calculate_v_rho0_i()", f"Rob: {robot_index} | No neighbours found, so v_rho0_i set to [0.0, 0.0, 0.0]")
		v_rho0_i = [0.0, 0.0, 0.0]
	else:
		p_rho_i = robots[robot_index].p_rho_i
		
		p_rho_ = []
		for j in range(len(N)):
			p_rho_.append(robots[N[j]].p_rho_i)


		term1 = (-1*c_1 / len(N)) * np.array(np.sum([np.multiply(np.sign([a - b for a,b in zip(p_rho_i, p_rho_[j])]) , np.absolute([a - b for a,b in zip(p_rho_i, p_rho_[j])]) ** alpha)
						for j in range(len(N))], axis=0))
		
		term2 = (1 / len(N)) * np.array(np.sum([robs_initial_v_rho0_i[N[j]] for j in range(len(N))], axis=0))
		# log("", f"v_rho0_i term1: {np.round(term1, decimals=2)} | term2: {np.round(term2, decimals=2)}", True)
		v_rho0_i = term1 + term2

	robs_initial_v_rho0_i[robot_index] = v_rho0_i 
	return v_rho0_i

def get_robot_target_rho(robot):
	curr_rho_x, curr_rho_y = robot.rho
	area = grey_grid[curr_rho_x-1:curr_rho_x+2, curr_rho_y-1:curr_rho_y+2]
	local_min = np.min(area)
	local_min_ind = np.unravel_index(area.argmin(), area.shape)
	target_rho = [curr_rho_x + local_min_ind[0] -1, curr_rho_y + local_min_ind[1] -1]
	
	if (local_min == 0) & (np.array_equal(local_min_ind,[1,1])):
		if show_log_get_robot_target_rho:
			log("get_robot_target_rho()", f"Robot {robot.index} inside shape")

	return target_rho

def get_robot_target_p_rho0(target_rho):
	target_p_rho_x = actual_environment_x_min + target_rho[0] * normalized_x_steps + normalized_x_steps / 2
	target_p_rho_y = actual_environment_y_min + target_rho[1] * normalized_y_steps + normalized_y_steps / 2

	# Center point (in positional meters) of the cell robot i is targeted to occupy
	target_p_rho_i = [target_p_rho_x, target_p_rho_y,0]
	
	return target_p_rho_i

def shape_entering_velocity(robots, robot_index):
	"""
	Calculate the shape-entering velocity component for a robot.
	
	Returns:
	- v_ent_i: The shape-entering velocity component
	"""

	# Selected the position of center of cell the robot is currently in
	p_i = robots[robot_index].p_rho_i
	
	p_t_i_ind = get_robot_target_rho(robots[robot_index])
	p_t_i = get_robot_target_p_rho0(p_t_i_ind)

	xi_rho_i = robots[robot_index].xi_rho
	v_rho0_i = calculate_v_rho0_i(robots, robot_index)

	top = ([a - b for a,b in zip(p_t_i, p_i)])
	bottom = np.linalg.norm([a - b for a,b in zip(p_t_i, p_i)])
	unit_vector = np.divide(top, bottom, out=np.zeros_like(top), where=bottom!=0)
	
	firstpart = k_1 * xi_rho_i * unit_vector
	secondpart = v_rho0_i
	v_ent_i = ([a + b for a,b in zip(firstpart, secondpart)])
	print(f"Rob: {robot_index} | Color: {xi_rho_i} | v_rho0: {np.round(v_rho0_i, decimals=2)}")
	return v_ent_i
