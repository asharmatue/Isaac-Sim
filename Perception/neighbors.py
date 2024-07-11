import numpy as np

from Settings.settings import num_robots, r_sense

def neighboring_i(robots, robot_index):
	base_robot = robots[robot_index]

	# # 1D List only containing Neighbour (Self not counted as Neighbour)
	N_list = []
	for other_robot_index in range(num_robots):
		if robot_index != other_robot_index:
			diff = ([a - b for a,b in zip(base_robot.pos, robots[other_robot_index].pos)])
			if (np.linalg.norm(diff) < r_sense):
				N_list.append(other_robot_index)

	return N_list
