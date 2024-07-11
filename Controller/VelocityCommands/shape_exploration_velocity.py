import numpy as np

from Perception.grid import grey_grid, normalized_x_steps, normalized_y_steps, number_of_rows, number_of_columns
from Perception.grid import get_pos_of_rho
from Settings.settings import num_robots, r_sense, r_body, r_avoid
from Settings.settings import show_log_in_shape_boundary, show_log_neighbouring_cells, show_log_shape_exploration_velocity
from Settings.util import log


def psi_weight(arg):
	if arg <= 0:
		return 1
	elif arg < 1:
		return 0.5 * (1 + np.cos(np.pi * arg))
	else:
		return 0

def in_shape_boundary(robot):
	# if: robot i is close to the boundary of the shape so that there are non-black cells within the sensing radius r_sense
		# return False
	# else:
		# return True 

	curr_rho_x, curr_rho_y = robot.rho
	in_shape = False
	
	# Radius r_sense means this number of cells: 
	r_sense_cell_x = np.int(r_sense/normalized_x_steps)
	r_sense_cell_y = np.int(r_sense/normalized_y_steps)
	
	# neighbouring cells within radius r_sense 
	area = grey_grid[curr_rho_x-r_sense_cell_x:curr_rho_x+r_sense_cell_x+1, curr_rho_y-r_sense_cell_y:curr_rho_y+r_sense_cell_y+1]
	
	# Find max value of neighbouring cells within radius r_sense 
	local_max_color = np.max(area)

	# If max value is black, then inside shape boundary, else not inside shape boundary
	if local_max_color > 0:
		in_shape = False
	elif local_max_color == 0:
		in_shape = True

	if show_log_in_shape_boundary:
		log("in_shape_boundary()", f"Rob: {robot.index} | in_shape?: {in_shape} | max_color: {np.round(local_max_color,2)} | r_sense_cells x:{r_sense_cell_x} y:{r_sense_cell_x} ")
		log("", f"area:\n{np.round(area,2)}\n")
	
	return in_shape, r_sense_cell_x, r_sense_cell_y, area

def occupied_cells(robots):
	# return rho of occupied cells, considering radius of robot r_body
	occupied = set()  # Using a set to avoid duplicate entries

	x_numcells = number_of_rows
	y_numcells = number_of_columns
	x_cellsize = normalized_x_steps
	y_cellsize = normalized_y_steps

	# Center cell coordinates
	center_cell_x = (x_numcells - 1) // 2 
	center_cell_y = (y_numcells - 1) // 2

	# Process each robot
	for robot_index in range(num_robots):
		robot_center_x, robot_center_y, _ = robots[robot_index].pos # Center of each robot

		# Calculate the indices of the grid that intersect the bounding box of the robot
		min_x = max(0, int((robot_center_x - r_body) / x_cellsize + center_cell_x))
		max_x = min(x_numcells - 1, int((robot_center_x + r_body) / x_cellsize + center_cell_x))
		min_y = max(0, int((robot_center_y - r_body) / y_cellsize + center_cell_y))
		max_y = min(y_numcells - 1, int((robot_center_y + r_body) / y_cellsize + center_cell_y))

		# Iterate over the cells within the bounding box
		for x in range(min_x, max_x + 1):
			for y in range(min_y, max_y + 1):
				# Calculate the real-world (x, y) coordinates of the center of each cell
				cell_x_center = (x - center_cell_x) * x_cellsize
				cell_y_center = (y - center_cell_y) * y_cellsize

				# Check if any part of the robot covers the center of the cell. Factor sqrt(2) added to cover edge cases if robot perfectly between 4 cells
				# Paper wants to count occupied if cell_center within r_avoid/2 of any robot. If want cells occupied by body: np.sqrt(2)*r_body
				if np.sqrt((cell_x_center - robot_center_x) ** 2 + (cell_y_center - robot_center_y) ** 2) <= r_avoid/2: 
					occupied.add((x, y))

	log("occupied_cells()", f"Occupied cells:\n {list(occupied)}\n")
	return list(occupied)

def neighbouring_cells(robots, robot_index):
	in_shape, r_sense_cell_x, r_sense_cell_y, area  = in_shape_boundary(robots[robot_index])
	M_cells = []

	curr_rho_x, curr_rho_y = robots[robot_index].rho

	if in_shape == False:
		for i in range(len(area)):
			for j in range(len(area)):
				# If neighbouring cell within radius r_sense is black, append it to M_cells 
				if area[i , j] == 0:
					M_cells.append([curr_rho_x-r_sense_cell_x+i , curr_rho_y-r_sense_cell_y+j])

	elif in_shape == True:
		
		occupied_cells = occupied_cells(robots)

		M_cells_debug = []

		for i in range(len(area)):
			for j in range(len(area)):
				# If neighbouring cell within radius r_sense is black...
				if area[i , j] == 0:
						# AND if neighbouring cell within radius r_sense is unoccupied, append it to M_cells
						M_cells_debug.append([curr_rho_x-r_sense_cell_x+i , curr_rho_y-r_sense_cell_y+j])
						if ((curr_rho_x-r_sense_cell_x+i, curr_rho_y-r_sense_cell_y+j) not in occupied_cells):
							M_cells.append([curr_rho_x-r_sense_cell_x+i , curr_rho_y-r_sense_cell_y+j])

	if show_log_neighbouring_cells:
		log("neighbouring_cells()", f"Rob: {robot_index} | include occupied?: {in_shape} |  M_cells: {M_cells}")
	return M_cells

def shape_exploration_velocity(robots, robot_index):
	# To optimize code, first check if any neighboring valid cells. If found, only then set parameters, call nessesary functions, and perform any calculations
	
	M_i_neigh = neighbouring_cells(robots, robot_index)
	
	if len(M_i_neigh) <= 0:
		if show_log_shape_exploration_velocity:
			log("shape_exploration_velocity()", f"No valid cells found, v_exp_i set to [0.0, 0.0]")
		v_exp_i = [0.0, 0.0]
		return v_exp_i
	
	# Hardcode
	sigma_1 = 10    # 10 or 5 in paper
	sigma_2 = 20    # 20 or 15 in paper
	in_shape, _, _, _  = in_shape_boundary(robots[robot_index])
	k_2 = sigma_1 if (not in_shape) else sigma_2

	p_rhos = []
	for j in range(len(M_i_neigh)):
		p_rhos.append(get_pos_of_rho(M_i_neigh[j]))
	
	p_i = robots[robot_index].pos[0:2]
	
	top = sum([k_2 * np.multiply(
						psi_weight(np.linalg.norm([a - b for a,b in zip(p_rhos[rho], p_i)]) / r_sense ) 
						, ([a - b for a,b in zip(p_rhos[rho], p_i)])    
					) 
			for rho in range(len(M_i_neigh))])
	
	bottom = sum([psi_weight( np.linalg.norm([a - b for a,b in zip(p_rhos[rho], p_i)]) / r_sense ) 
				for rho in range(len(M_i_neigh))])
	
	v_exp_i = np.divide(np.array(top), np.array(bottom), out=np.zeros_like(top), where=bottom!=0.0)
		
	return v_exp_i
