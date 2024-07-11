from omni.isaac.examples.base_sample import BaseSample

import numpy as np
from omni.isaac.examples.user_examples.git_isaac_sim.directory import directory_setup
directory_setup()

import omni                                                     # Provides the core omniverse apis
import asyncio                                                  # Used to run sample asynchronously to not block rendering thread
from omni.isaac.range_sensor import _range_sensor               # Imports the python bindings to interact with lidar sensor

from pxr import UsdShade, Sdf, Gf, Vt, UsdPhysics
from omni.isaac.core.materials import OmniPBR
import omni.isaac.core.utils.prims as prims_utils
from omni.isaac.core.physics_context import PhysicsContext

from controllers import DiffDriveController
from omni.isaac.wheeled_robots.controllers.wheel_base_pose_controller import WheelBasePoseController
from omni.isaac.wheeled_robots.controllers.differential_controller import DifferentialController
from omni.isaac.wheeled_robots.controllers.holonomic_controller import HolonomicController

stage = omni.usd.get_context().get_stage()                      # Used to access Geometry
timeline = omni.timeline.get_timeline_interface()               # Used to interact with simulation
lidarInterface = _range_sensor.acquire_lidar_sensor_interface() # Used to interact with the LIDAR

from settings import num_robots, input_shape, h, r_avoid, r_sense, r_check, r_body
from settings import entering_weight, exploration_weight, interaction_weight
from settings import c_1, alpha, k_1, k_3, forward_gain, angle_gain
from settings import actual_environment_size_x, actual_environment_size_y, actual_environment_x_min, actual_environment_x_max, actual_environment_y_min, actual_environment_y_max
from settings import show_vel_spheres, show_robot_obstacle_positions
from settings import remove_redundant_obstacle_positions
from settings import EXPORT_DATA
import data_export
from grid import number_of_rows, number_of_columns, normalized_x_steps, normalized_y_steps
from grid import grey_grid, get_grid_rho, get_xi_rho, get_pos_of_rho
from environment_setup import setup_environment
from robot_setup import setup_robots, base_sphere_prim_path, base_sphere_prim_path_suffix
from util import log, performance_timestamp, mod
from robot import Robot
from communicator import send_robot_data

from Perception.lidar import get_lidar


from omni.isaac.core.objects import VisualSphere
from settings import show_log_velocity_commands, show_log_get_robot_target_rho, show_log_find_collision_points, show_interaction_velocity, show_log_in_shape_boundary, show_log_neighbouring_cells, show_log_shape_exploration_velocity, show_log_send_robot_actions

# For Obstacle Collision Point Visualisation Spheres in find_colliion_points()
obs_counter = 0
highest_obs_index = [0 for _ in range(num_robots)]

# For Velocity Visualisation Spheres in velocity_commands()
mtl_created_list = []
spheres_prim = [[0,0,0] for _ in range(num_robots)]
spheres_mat = [[0,0,0] for _ in range(num_robots)]
mtl_prim = [[0,0,0] for _ in range(num_robots)] 

# Inital v_rho0_i set to 0
robs_initial_v_rho0_i = [[0,0,0] for _ in range(num_robots)]

class Main(BaseSample):

    robots:list[Robot]
    '''List of robots in the simulation world'''

    def __init__(self) -> None:
        super().__init__()
        self.v_rho0_cache = {}
        if (EXPORT_DATA):
            data_export.clearData()
        return
    
    def setup_scene(self):
        self.world = self.get_world()
        setup_environment(self.world) 
        setup_robots(self.world)
        
        if show_vel_spheres:
            for _ in range(num_robots):
                for _ in range(3):
                    omni.kit.commands.execute(
                        "CreateAndBindMdlMaterialFromLibrary",
                        mdl_name="OmniPBR.mdl",
                        mtl_name=f"OmniPBR",
                        mtl_created_list=mtl_created_list,
                    )
        
    async def setup_post_load(self):
        self._world = self.get_world()
        self.robots = [None for _ in range(num_robots)]
        for robot_index in range(num_robots):
            self.robots[robot_index] = Robot(self._world, robot_index)

        data_export.exportHeader() # After adding export values in Robot class.
            
        self._world.add_physics_callback("sending_actions", callback_fn=self.send_robot_actions)
        # Initialize our controller after load and the first reset
        self._Vel_controller = DiffDriveController()
        self._WBP_controller = WheelBasePoseController(name="wheel_base_pose_controller",
                                                        open_loop_wheel_controller=
                                                            DifferentialController(name="diff_controller",
                                                                                    wheel_radius=0.03, wheel_base=0.1125),
                                                    is_holonomic=False)
        
        # Changing inital v_rho0_i from [0,0,0] to become get_robot_vel() initially. 
        # Afterwards these values are overwritten in calculate_v_rho0_i() as intented
        for robot_index in range(num_robots):
            robs_initial_v_rho0_i[robot_index] = self.robots[robot_index].vel
        return
    
    def velocity_commands(self, robot_index):
        raw_entering = self.shape_entering_velocity(robot_index)[0:2]
        raw_exploration = self.shape_exploration_velocity(robot_index)
        raw_interaction = self.interaction_velocity(robot_index)[0:2]
        
        applied_entering = np.multiply(entering_weight, raw_entering) 
        applied_exploration = np.multiply(exploration_weight, raw_exploration)
        applied_interaction = np.multiply(interaction_weight, raw_interaction)
        
        v = []
        v.append(applied_entering)
        v.append(applied_exploration)
        v.append(applied_interaction)
        v_i = np.sum([v[j] for j in range(len(v))], axis=0)
        
        if show_log_velocity_commands:

            log("velocity_commands()", f"Rob: {robot_index} | Velocity commands: {np.round(v_i, decimals=2)}")
            log("individual raw", f"Ent:{np.round(raw_entering, decimals=2)} | Exp:{np.round(raw_exploration, decimals=2)} | Int:{np.round(raw_interaction, decimals=2)}", True)
            log("individual weighted", f"Ent:{np.round(v[0], decimals=2)} | Exp:{np.round(v[1], decimals=2)} | Int:{np.round(v[2], decimals=2)}", True)

        if show_vel_spheres:
            # performance_timestamp("")
            entering_mag = np.linalg.norm(v[0])
            exploration_mag = np.linalg.norm(v[1])
            interaction_mag = np.linalg.norm(v[2])
            total_mag = entering_mag + exploration_mag + interaction_mag

            entering_color = Gf.Vec3f((entering_mag/total_mag), 0.0, 0.0)
            exploration_color = Gf.Vec3f(0.0, (exploration_mag/total_mag), 0.0)
            interaction_color = Gf.Vec3f(0.0, 0.0, (interaction_mag/total_mag))
            vel_colors = [entering_color, exploration_color, interaction_color]
            if show_log_velocity_commands:
                log("induvidual color %", f"Ent (R): {np.round(vel_colors[0][0]*50, decimals=2)}% | Exp (G): {np.round(vel_colors[1][1]*50, decimals=2)}% | Int (B): {np.round(vel_colors[2][2]*50, decimals=2)}%", True)
            
            stage = omni.usd.get_context().get_stage()

            # Method 1 - Each prim and material is indexed by robot index and velocity command, so each sphere has unique id
            # vel: 0 ent, 1 exp, 2 int
            for vel in range(3):
                mtl_prim[robot_index][vel] = stage.GetPrimAtPath(mtl_created_list[robot_index*3 + vel])
                omni.usd.create_material_input(mtl_prim[robot_index][vel], "diffuse_color_constant", vel_colors[vel], Sdf.ValueTypeNames.Color3f)
                path = f"{base_sphere_prim_path}{robot_index:02}/chassis{base_sphere_prim_path_suffix[vel]}{robot_index:02}"
                spheres_prim[robot_index][vel] = stage.GetPrimAtPath(path)
                spheres_mat[robot_index][vel] = UsdShade.Material(mtl_prim[robot_index][vel])
                UsdShade.MaterialBindingAPI(spheres_prim[robot_index][vel]).Bind(spheres_mat[robot_index][vel], UsdShade.Tokens.strongerThanDescendants)
            # performance_timestamp("Set colors")   
        return v_i

    def neighboring_i(self, robot_index):
        base_robot = self.robots[robot_index]

        # # 1D List only containing Neighbour (Self not counted as Neighbour)
        N_list = []
        for other_robot_index in range(num_robots):
            if robot_index != other_robot_index:
                diff = ([a - b for a,b in zip(base_robot.pos, self.robots[other_robot_index].pos)])
                if (np.linalg.norm(diff) < r_sense):
                    N_list.append(other_robot_index)

        return N_list

    def calculate_v_rho0_i(self, robot_index):
        """
        Calculate the local interpretation of the moving velocity of the entire shape.
        Returns:
        - v_rho_i: The local interpretation of the moving velocity of the entire shape for robot i.
        """

        N = self.neighboring_i(robot_index)

        if len(N) == 0:
            log("calculate_v_rho0_i()", f"Rob: {robot_index} | No neighbours found, so v_rho0_i set to [0.0, 0.0, 0.0]")
            v_rho0_i = [0.0, 0.0, 0.0]
        else:
            p_rho_i = self.robots[robot_index].p_rho_i
            
            p_rho_ = []
            for j in range(len(N)):
                p_rho_.append(self.robots[N[j]].p_rho_i)


            term1 = (-1*c_1 / len(N)) * np.array(np.sum([np.multiply(np.sign([a - b for a,b in zip(p_rho_i, p_rho_[j])]) , np.absolute([a - b for a,b in zip(p_rho_i, p_rho_[j])]) ** alpha)
                            for j in range(len(N))], axis=0))
            
            term2 = (1 / len(N)) * np.array(np.sum([robs_initial_v_rho0_i[N[j]] for j in range(len(N))], axis=0))
            # log("", f"v_rho0_i term1: {np.round(term1, decimals=2)} | term2: {np.round(term2, decimals=2)}", True)
            v_rho0_i = term1 + term2

        robs_initial_v_rho0_i[robot_index] = v_rho0_i 
        return v_rho0_i

    def get_robot_target_rho(self, robot_index):
        curr_rho_x, curr_rho_y = self.robots[robot_index].rho
        area = grey_grid[curr_rho_x-1:curr_rho_x+2, curr_rho_y-1:curr_rho_y+2]
        local_min = np.min(area)
        local_min_ind = np.unravel_index(area.argmin(), area.shape)
        target_rho = [curr_rho_x + local_min_ind[0] -1, curr_rho_y + local_min_ind[1] -1]
        
        if (local_min == 0) & (np.array_equal(local_min_ind,[1,1])):
            if show_log_get_robot_target_rho:
                log("get_robot_target_rho()", f"Robot {robot_index} inside shape")

        return target_rho

    def get_robot_target_p_rho0(self, target_rho):
        target_p_rho_x = actual_environment_x_min + target_rho[0] * normalized_x_steps + normalized_x_steps / 2
        target_p_rho_y = actual_environment_y_min + target_rho[1] * normalized_y_steps + normalized_y_steps / 2

        # Center point (in positional meters) of the cell robot i is targeted to occupy
        target_p_rho_i = [target_p_rho_x, target_p_rho_y,0]
        
        return target_p_rho_i

    def shape_entering_velocity(self, robot_index):
        """
        Calculate the shape-entering velocity component for a robot.
        
        Returns:
        - v_ent_i: The shape-entering velocity component
        """

        # Selected the position of center of cell the robot is currently in
        p_i = self.robots[robot_index].p_rho_i
        
        p_t_i_ind = self.get_robot_target_rho(robot_index)
        p_t_i = self.get_robot_target_p_rho0(p_t_i_ind)

        xi_rho_i = self.robots[robot_index].xi_rho
        v_rho0_i = self.calculate_v_rho0_i(robot_index)

        top = ([a - b for a,b in zip(p_t_i, p_i)])
        bottom = np.linalg.norm([a - b for a,b in zip(p_t_i, p_i)])
        unit_vector = np.divide(top, bottom, out=np.zeros_like(top), where=bottom!=0)
        
        firstpart = k_1 * xi_rho_i * unit_vector
        secondpart = v_rho0_i
        v_ent_i = ([a + b for a,b in zip(firstpart, secondpart)])
        print(f"Rob: {robot_index} | Color: {xi_rho_i} | v_rho0: {np.round(v_rho0_i, decimals=2)}")
        return v_ent_i

    def mu_weight(self, arg):
        mu = 0
        if arg <= r_check:
            mu = (r_check/arg) - 1
        else:
            mu = 0
        return mu

    def find_collision_points_index(self, robot_index):
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
        
    def find_collision_points(self, robot_index):
        coll_ind = np.array(self.find_collision_points_index(robot_index))
        obstacle_pos = []
        if not coll_ind.size:           # If no walls, return empty array # Same as if len(coll_ind) <= 0
            if show_log_find_collision_points: 
                log("find_collision_points()", "No collision points found, output set to []")
            return np.array(obstacle_pos)
        
        # If walls only then do calculations
        distance, angle = get_lidar(robot_index)
        curr_pos = self.robots[robot_index].pos
        curr_ori = self.robots[robot_index].euler_ori

        if coll_ind.size == 1: # Added to prevent error where sometimes coll_ind does exist but is a float instead of an array of size 1
            obstacle_pos.append( [ float(curr_pos[0] + distance[coll_ind]*np.cos(curr_ori[2] + angle[coll_ind])) , float(curr_pos[1] + distance[coll_ind]*np.sin(curr_ori[2] + angle[coll_ind])) , 0.0 ] )
        else:
            for i in range(coll_ind.size): # Should work same as for i in range(len(coll_ind)):
                obstacle_pos.append( [ float(curr_pos[0] + distance[coll_ind[i]]*np.cos(curr_ori[2] + angle[coll_ind[i]])) , float(curr_pos[1] + distance[coll_ind[i]]*np.sin(curr_ori[2] + angle[coll_ind[i]])) , 0.0 ] )    
        # performance_timestamp("")
        
        # Modifier toggled in settings.py
        if remove_redundant_obstacle_positions:
            show_log_remove_redundant_obstacle_positions = False
            robot_pos = []
            for other_robot_index in range(num_robots):
                if other_robot_index != robot_index:
                    robot_pos.append(self.robots[other_robot_index].pos)
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
        
        # Visualisation 
        if show_robot_obstacle_positions:

            world = self.get_world()
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
        if show_log_find_collision_points:
            log("find_collision_points()", f"Obstacles:\n{np.array(obstacle_pos).round(decimals=2)}")
        # performance_timestamp("End of f_c_p")
        return np.array(obstacle_pos)

    def interaction_velocity(self, robot_index):
        
        N = self.neighboring_i(robot_index)
        O = self.find_collision_points(robot_index)
        
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
                    p.append(self.robots[N[j]].pos) 
            else:                           # If both O and N, change length of sum, length_sum, and collision point positions, p.
                if show_interaction_velocity:
                    log("interaction_velocity()", f"Neighbours AND Collision points found")
                length_sum = len(N)+len(O)-1
                for j in range(length_sum):
                    if j < len(N):
                        p.append(self.robots[N[j]].pos)
                    else:
                        p.append(O[j-len(N)])

            
            v_i = self.robots[robot_index].vel
            if v_i.size <= 1:
                v_i = np.array([0.0 , 0.0 , 0.0])
            v_ = []
            for j in range(len(N)):
                v_.append(self.robots[N[j]].vel)
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
            p_i = self.robots[robot_index].pos # get_robot_p_rho0

            term1 = np.array(np.sum([
                                    np.multiply(
                                        self.mu_weight( 
                                            np.linalg.norm( [a - b for a,b in zip(p_i, p[j])] ) 
                                        ) 
                                        , [a - b for a,b in zip(p_i, p[j])]
                                    )
                            for j in range(length_sum)], axis=0)
                    )
        
        firstterm = np.multiply(term1, k_3)
        v_int_i = ([a - b for a,b in zip(firstterm, term2)])
        return np.array(v_int_i)

    def psi_weight(self, arg):
        if arg <= 0:
            return 1
        elif arg < 1:
            return 0.5 * (1 + np.cos(np.pi * arg))
        else:
            return 0

    def in_shape_boundary(self, robot_index):
        # if: robot i is close to the boundary of the shape so that there are non-black cells within the sensing radius r_sense
            # return False
        # else:
            # return True 

        curr_rho_x, curr_rho_y = self.robots[robot_index].rho
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
            log("in_shape_boundary()", f"Rob: {robot_index} | in_shape?: {in_shape} | max_color: {np.round(local_max_color,2)} | r_sense_cells x:{r_sense_cell_x} y:{r_sense_cell_x} ")
            log("", f"area:\n{np.round(area,2)}\n")
        
        return in_shape, r_sense_cell_x, r_sense_cell_y, area

    def occupied_cells(self):
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
            robot_center_x, robot_center_y, _ = self.robots[robot_index].pos # Center of each robot

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
    
    def neighbouring_cells(self, robot_index):
        in_shape, r_sense_cell_x, r_sense_cell_y, area  = self.in_shape_boundary(robot_index)
        M_cells = []

        curr_rho_x, curr_rho_y = self.robots[robot_index].rho

        if in_shape == False:
            for i in range(len(area)):
                for j in range(len(area)):
                    # If neighbouring cell within radius r_sense is black, append it to M_cells 
                    if area[i , j] == 0:
                        M_cells.append([curr_rho_x-r_sense_cell_x+i , curr_rho_y-r_sense_cell_y+j])

        elif in_shape == True:
            
            occupied_cells = self.occupied_cells()

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
    
    def shape_exploration_velocity(self, robot_index):
        # To optimize code, first check if any neighboring valid cells. If found, only then set parameters, call nessesary functions, and perform any calculations
        
        M_i_neigh = self.neighbouring_cells(robot_index)
        
        if len(M_i_neigh) <= 0:
            if show_log_shape_exploration_velocity:
                log("shape_exploration_velocity()", f"No valid cells found, v_exp_i set to [0.0, 0.0]")
            v_exp_i = [0.0, 0.0]
            return v_exp_i
        
        # Hardcode
        sigma_1 = 10    # 10 or 5 in paper
        sigma_2 = 20    # 20 or 15 in paper
        in_shape, _, _, _  = self.in_shape_boundary(robot_index)
        k_2 = sigma_1 if (not in_shape) else sigma_2
    
        p_rhos = []
        for j in range(len(M_i_neigh)):
            p_rhos.append(get_pos_of_rho(M_i_neigh[j]))
        
        p_i = self.robots[robot_index].pos[0:2]
        
        top = sum([k_2 * np.multiply(
                            self.psi_weight(np.linalg.norm([a - b for a,b in zip(p_rhos[rho], p_i)]) / r_sense ) 
                            , ([a - b for a,b in zip(p_rhos[rho], p_i)])    
                        ) 
                for rho in range(len(M_i_neigh))])
        
        bottom = sum([self.psi_weight( np.linalg.norm([a - b for a,b in zip(p_rhos[rho], p_i)]) / r_sense ) 
                    for rho in range(len(M_i_neigh))])
        
        v_exp_i = np.divide(np.array(top), np.array(bottom), out=np.zeros_like(top), where=bottom!=0.0)
         
        return v_exp_i

    def send_robot_actions(self, step_size):
        for robot in self.robots:
            robot.update()
            send_robot_data(robot)

        data_export.exportData()


        for robot_index in range(num_robots): 
            v_x, v_y = self.velocity_commands(robot_index) 
            curr_rot = self.robots[robot_index].euler_ori
            
            forward_raw = (((v_x ** 2) + (v_y ** 2)) ** 0.5)
            angle_raw = mod((np.rad2deg(np.arctan2(v_y,v_x) - curr_rot[2]) + 180) , 360) - 180  #degrees      
            
            # Rotation Velocity Compensation: If turning by big amount, forward velocity is smaller
            # Returns a weight between 0 and 1. If needing to turn 180 degrees, weight 0; If turning 0 degrees, weight = 1
            rotation_compensation = True
            if rotation_compensation:
                rotation_compensation_1 = ((180 - np.abs(angle_raw)) % 180) / 180  

                rotation_compensation_angle = 1 #(1 + 0.5*(1-rotation_compensation_1))

                # Directly compare effect in print log statement
                no_rot_comp_angle = angle_gain * np.deg2rad(angle_raw)
                no_rot_comp_forward = forward_gain * forward_raw
            else:
                rotation_compensation_angle = 1
                rotation_compensation_1 = 1

            angle =  rotation_compensation_angle * angle_gain * np.deg2rad(angle_raw)
            forward = rotation_compensation_1 * forward_gain * forward_raw

            self.robots[robot_index].instance.apply_action(self._Vel_controller.forward(command=[forward, angle]))
            if show_log_send_robot_actions:
                log("send_robot_actions()", f"Rob: {robot_index} | Velocities forward: {np.round(forward, decimals=2)} m/s | angular: {np.round(angle, decimals=2)} rads/s")
                log("", f"Raw velocities forward: {np.round(forward_raw, decimals=2)} m/s | angular: {np.round(angle_raw, decimals=2)} deg/s", True)
                
                if rotation_compensation:
                    log("", f"Target direction: {np.rad2deg(np.arctan2(v_y,v_x)).round(decimals=2)}, Current direction: {np.rad2deg(curr_rot[2]).round(decimals=2)}, Difference: {np.rad2deg(np.arctan2(v_y,v_x) - curr_rot[2]).round(decimals=2)}")
                    # log("", f"Without rot comp: {np.round(no_rot_comp_angle,decimals=2)} rad/s, {np.round(no_rot_comp_forward,decimals=2)} m/s | With: {np.round(angle, decimals=2)} rad/s, {np.round(forward, decimals=2)} m/s", True)
                    log("", f"Rot comp angle multiplier: {np.round(rotation_compensation_angle, decimals=2)} , forward multiplier: {np.round(rotation_compensation_1,decimals=2)}", True)
                    
        return
     