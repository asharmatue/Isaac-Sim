import omni
import numpy as np

import omni.isaac.core.utils.prims as prims_utils
from omni.isaac.core.objects import VisualSphere

from Settings.settings import num_robots, r_check, r_body, k_3
from Settings.settings import REMOVE_REDUNDANT_OBSTACLE_POSITIONS, SHOW_ROBOT_OBSTACLE_POSITIONS, show_log_find_collision_points, show_interaction_velocity
from Settings.util import log
from Perception.lidar import get_lidar
from Perception.neighbors import neighboring_i

# For Obstacle Collision Point Visualisation Spheres in find_colliion_points()
obs_counter = 0
highest_obs_index = [0 for _ in range(num_robots)]

def mu_weight(arg):
	mu = 0
	if arg <= r_check:
		mu = (r_check/arg) - 1
	else:
		mu = 0
	return mu

def find_collision_points_index(robot_index):
	distance, angle = get_lidar(robot_index)
	wall_indices = []
	end_wall_index = []
	current_end_of_wall_index = -1
	
	# Copy relevant indicies into wall_indices and store end wall indicies
	for i in range(len(distance)):
		if distance[i] < r_check:
			wall_indices.append(i)
			current_end_of_wall_index = i
		elif current_end_of_wall_index != -1:
			end_wall_index.append(current_end_of_wall_index)
			current_end_of_wall_index = -1
	
	if (distance[0] >= r_check and current_end_of_wall_index != -1):
		end_wall_index.append(current_end_of_wall_index)
	
	if len(wall_indices) <= 0:          # If no walls, no calculations needed return empty list, wall_indices = [] 
		log("find_collision_points_index()", f"No collision index found, wall_indicies: {wall_indices} output set to []:{np.array([])}")
		return np.array([])
	else:                               # If walls, do calculations
		
		collision_points_index = []

		# Find closest point of walls
		if len(end_wall_index) > 0:
			# Find closest point on every wall
			for i in range(len(end_wall_index)):
				if i == 0:
					if distance[wall_indices[0]:end_wall_index[0]].size: # Added to solve "attempt to get argmin of an empty sequence" error which happens sometimes
						# Find the index of the minimum distance to robot from of all wall_indices of first wall and add it to collision_points_index
						collision_points_index.append(np.argmin(np.array(distance[wall_indices[0]:end_wall_index[0]]))) 
				else:
					# Find the index of the minimum distance to robot from of all wall_indices of following walls + the index shift and add it to collision_points_index
					collision_points_index.append(np.argmin(np.array(distance[end_wall_index[i-1]+1:end_wall_index[i]])) + end_wall_index[i-1]+1) 
		# Find closest point of the 1 (one) wall
		elif len(wall_indices) > 0:
			# Find the index of the minimum distance to robot from of all wall_indices of the wall
			collision_points_index.append(wall_indices[np.argmin(np.array(distance[wall_indices[0]:wall_indices[-1]]))]) # -1 is last index
	
		return np.array(collision_points_index)
	
def find_collision_points(robots, robot_index, world):
	coll_ind = np.array(find_collision_points_index(robot_index))
	obstacle_pos = []
	if not coll_ind.size:           # If no walls, return empty array # Same as if len(coll_ind) <= 0
		if show_log_find_collision_points: 
			log("find_collision_points()", "No collision points found, output set to []")
		return np.array(obstacle_pos)
	
	# If walls only then do calculations
	distance, angle = get_lidar(robot_index)
	curr_pos = robots[robot_index].pos
	curr_ori = robots[robot_index].euler_ori

	if coll_ind.size == 1: # Added to prevent error where sometimes coll_ind does exist but is a float instead of an array of size 1
		obstacle_pos.append( [ float(curr_pos[0] + distance[coll_ind]*np.cos(curr_ori[2] + angle[coll_ind])) , float(curr_pos[1] + distance[coll_ind]*np.sin(curr_ori[2] + angle[coll_ind])) , 0.0 ] )
	else:
		for i in range(coll_ind.size): # Should work same as for i in range(len(coll_ind)):
			obstacle_pos.append( [ float(curr_pos[0] + distance[coll_ind[i]]*np.cos(curr_ori[2] + angle[coll_ind[i]])) , float(curr_pos[1] + distance[coll_ind[i]]*np.sin(curr_ori[2] + angle[coll_ind[i]])) , 0.0 ] )    
	# performance_timestamp("")
	
	remove_redundant_obstacle_positions(robots, robot_index, obstacle_pos)
	
	show_robot_obstacle_positions(robot_index, obstacle_pos, world)
	
	if show_log_find_collision_points:
		log("find_collision_points()", f"Obstacles:\n{np.array(obstacle_pos).round(decimals=2)}")
	# performance_timestamp("End of f_c_p")
	return np.array(obstacle_pos)

def remove_redundant_obstacle_positions(robots, robot_index, obstacle_pos):
	# Modifier toggled in settings.py
	if not REMOVE_REDUNDANT_OBSTACLE_POSITIONS:
		return
	
	show_log_remove_redundant_obstacle_positions = False
	robot_pos = []
	for other_robot_index in range(num_robots):
		if other_robot_index != robot_index:
			robot_pos.append(robots[other_robot_index].pos)
	if show_log_remove_redundant_obstacle_positions:
		print(f"Rob: {robot_index} Obstacles:\n{np.array(obstacle_pos).round(decimals=2)}")
	# performance_timestamp("get robot positions")
	# Hardcode
	safety_distance = 0.05
	indicies_to_delete = [] 
	for obstacle_index in range(len(obstacle_pos)):
		for other_robot_index in range(len(robot_pos)):
			dist = np.linalg.norm(np.subtract(robot_pos[other_robot_index], obstacle_pos[obstacle_index]))
			if  dist < (np.sqrt(2))*r_body + safety_distance:
				if show_log_remove_redundant_obstacle_positions:
					print(f"Redundant position: {np.round(obstacle_pos[obstacle_index],decimals=2)}, index: {obstacle_index}. Too close to robot position: {np.round(robot_pos[other_robot_index],decimals=2)}, distance: {np.round(dist,decimals=2)}m")
				indicies_to_delete.append(obstacle_index)
	# performance_timestamp("check obstacle position at robots")
	if np.array(indicies_to_delete).size > 0:               # Only attempt to delete if indicies_to_delete not empty list
		for obstacle_index in range(len(indicies_to_delete)-1, -1, -1):  # Have to go in reverse order to ensure correct values deleted
			del obstacle_pos[indicies_to_delete[obstacle_index]]
		if show_log_remove_redundant_obstacle_positions:
			print(f"Redundant position(s) {indicies_to_delete} deleted, updated Rob {robot_index} Obstacles:\n{np.array(obstacle_pos).round(decimals=2)}")
	# performance_timestamp("delete obstacle positions at robot")

def show_robot_obstacle_positions(robot_index, obstacle_pos, world):
	if not SHOW_ROBOT_OBSTACLE_POSITIONS:
		return
	
	base_obs_sphere_prim_path = f"/World/Obstacles/Obstacles_"
	base_obs_sphere_name = f"Obstacle"
	
	remove_unnessesary_obs_spheres = True       # True: 4.5 fps False: 11-12 fps
	if remove_unnessesary_obs_spheres:
		global highest_obs_index
		print(f"highest_obs_index[robot_index]: {highest_obs_index[robot_index]}")
		if prims_utils.get_prim_at_path(f"{base_obs_sphere_prim_path}{robot_index:02}_{np.int(highest_obs_index[robot_index]):02}").IsValid():
			prims_utils.delete_prim(f"{base_obs_sphere_prim_path}{robot_index:02}_{np.int(highest_obs_index[robot_index]):02}")
			if highest_obs_index[robot_index] > 0:
				highest_obs_index[robot_index] -= 1
				print(f"Updated lower highest_obs_index | Rob: {robot_index}, Highest index: {highest_obs_index[robot_index]}")
	# performance_timestamp("remove_unnessesary_obs_spheres")
	colors = []
	colors.append([1.0, 0.0, 0.0])      # Robot 0
	colors.append([0.0, 1.0, 0.0])      # Robot 1
	colors.append([0.0, 0.0, 1.0])      # Robot 2
	global obs_counter
	
	for j in range(len(obstacle_pos)):
		ox, oy, _ = obstacle_pos[j]
		world.scene.add(
			VisualSphere(
					prim_path=f"{base_obs_sphere_prim_path}{robot_index:02}_{j:02}",
					name=f"{base_obs_sphere_name}{obs_counter}",
					translation=np.array([ox, oy, 0.15]),
					scale=np.array([0.02, 0.02, 0.02]),  
					color=np.array(colors[robot_index])
				))
		obs_counter += 1
		print(f"obs_counter: {obs_counter}")
		# performance_timestamp("create spheres")
		if remove_unnessesary_obs_spheres:
			if j > highest_obs_index[robot_index]:
				highest_obs_index[robot_index] = j
				print(f"Updated higher highest_obs_index | Rob: {robot_index}, Highest index: {highest_obs_index[robot_index]}")
		# performance_timestamp("update highest obs index")

def interaction_velocity(robots, robot_index, world):
	
	N = neighboring_i(robots, robot_index)
	O = find_collision_points(robots, robot_index, world)
	
	if (len(O) <= 0) and (len(N) <= 0):     # Neither N nor O so no calculations needed return [0.0, 0.0, 0.0]
		if show_interaction_velocity:
			log("interaction_velocity()", f"Neither Neighbours NOR Collision points, v_int_i set to [0.0, 0.0, 0.0]")
		v_int_i = [0.0, 0.0, 0.0]
		return np.array(v_int_i)
	
	# At least one of N exist O so calculate values
	p = []
	
	if (len(N) <= 0):               # If no N, only calculate term1 and set term2 to [0.0, 0.0, 0.0]
		if show_interaction_velocity:
			log("interaction_velocity()", f"No Neighbours, only Collision points found, v_int_i term2 set to [0.0, 0.0, 0.0]")
		length_sum = len(O) # used to be len(O)-1 
		for j in range(length_sum):
			p.append(O[j]) 
		term2 = [0.0, 0.0, 0.0]
	else:                           # If N, calculate both term1 and term2
		if (len(O) <= 0):               # If no O and only N, change length of sum, length_sum, and collision point positions, p.
			if show_interaction_velocity:
				log("interaction_velocity()", f"Only Neighbours, no Collision points")
			length_sum = len(N)-1
			for j in range(length_sum):
				p.append(robots[N[j]].pos) 
		else:                           # If both O and N, change length of sum, length_sum, and collision point positions, p.
			if show_interaction_velocity:
				log("interaction_velocity()", f"Neighbours AND Collision points found")
			length_sum = len(N)+len(O)-1
			for j in range(length_sum):
				if j < len(N):
					p.append(robots[N[j]].pos)
				else:
					p.append(O[j-len(N)])

		
		v_i = robots[robot_index].vel
		if v_i.size <= 1:
			v_i = np.array([0.0 , 0.0 , 0.0])
		v_ = []
		for j in range(len(N)):
			v_.append(robots[N[j]].vel)
		v_ = np.array(v_)
		if v_.size <= 2:
			v_ = np.array([0.0 , 0.0 , 0.0])
		
		term2 = np.array(np.sum([
							np.multiply(
								(1 / len(N)) , np.subtract(v_i, v_[j]) #[a - b for a,b in zip(v_i, v_[j])]
							)
						for j in range(len(N))], axis=0)
						)

	p = np.array(p)

	if len(p) <= 0:         # If error with find_collision_points_index, set term1 to [0.0, 0.0, 0.0]
		if show_interaction_velocity:
			log("interaction_velocity()", f"Error with calculating term1, v_int_i term1 set to [0.0, 0.0, 0.0]")
		term1 = [0.0, 0.0, 0.0]
	else:                   # If no error calculate term1
		# Selected the position of center of cell the robot is currently in
		p_i = robots[robot_index].pos # get_robot_p_rho0

		term1 = np.array(np.sum([
								np.multiply(
									mu_weight( 
										np.linalg.norm( [a - b for a,b in zip(p_i, p[j])] ) 
									) 
									, [a - b for a,b in zip(p_i, p[j])]
								)
						for j in range(length_sum)], axis=0)
				)
	
	firstterm = np.multiply(term1, k_3)
	v_int_i = ([a - b for a,b in zip(firstterm, term2)])
	return np.array(v_int_i)
