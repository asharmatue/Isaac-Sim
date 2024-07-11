import numpy as np
import omni
from pxr import Gf, Sdf, UsdShade

from omni.isaac.wheeled_robots.controllers.wheel_base_pose_controller import WheelBasePoseController
from omni.isaac.wheeled_robots.controllers.differential_controller import DifferentialController

from Settings.settings import num_robots, entering_weight, exploration_weight, interaction_weight, angle_gain, forward_gain
from Settings.settings import show_log_velocity_commands, SHOW_VEL_SPHERES, show_log_send_robot_actions
from Robot.robot_setup import base_sphere_prim_path, base_sphere_prim_path_suffix
from Controller.controllers import DiffDriveController
from Controller.VelocityCommands.shape_entering_velocity import shape_entering_velocity, initialize_v_rho0_i
from Controller.VelocityCommands.shape_exploration_velocity import shape_exploration_velocity
from Controller.VelocityCommands.interaction_velocity import interaction_velocity
from Settings.util import mod, log

velocity_controller = DiffDriveController()

# For Velocity Visualisation Spheres in velocity_commands()
mtl_created_list = []
spheres_prim = [[0,0,0] for _ in range(num_robots)]
spheres_mat = [[0,0,0] for _ in range(num_robots)]
mtl_prim = [[0,0,0] for _ in range(num_robots)] 

def velocity_commands(robots, robot_index, world):
	raw_entering = shape_entering_velocity(robots, robot_index)[0:2]
	raw_exploration = shape_exploration_velocity(robots, robot_index)
	raw_interaction = interaction_velocity(robots, robot_index, world)[0:2]
	
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

		# performance_timestamp("")

	show_vel_spheres(robot_index, v)

	return v_i

def show_vel_spheres(robot_index, v):
	if not SHOW_VEL_SPHERES:
		return
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

def apply_velocities(robots, world):

	for robot in robots:
            v_x, v_y = velocity_commands(robots, robot.index, world) 
            curr_rot = robot.euler_ori
            
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

            
            robot.instance.apply_action(velocity_controller.forward(command=[forward, angle]))
            if show_log_send_robot_actions:
                log("send_robot_actions()", f"Rob: {robot.index} | Velocities forward: {np.round(forward, decimals=2)} m/s | angular: {np.round(angle, decimals=2)} rads/s")
                log("", f"Raw velocities forward: {np.round(forward_raw, decimals=2)} m/s | angular: {np.round(angle_raw, decimals=2)} deg/s", True)
                
                if rotation_compensation:
                    log("", f"Target direction: {np.rad2deg(np.arctan2(v_y,v_x)).round(decimals=2)}, Current direction: {np.rad2deg(curr_rot[2]).round(decimals=2)}, Difference: {np.rad2deg(np.arctan2(v_y,v_x) - curr_rot[2]).round(decimals=2)}")
                    # log("", f"Without rot comp: {np.round(no_rot_comp_angle,decimals=2)} rad/s, {np.round(no_rot_comp_forward,decimals=2)} m/s | With: {np.round(angle, decimals=2)} rad/s, {np.round(forward, decimals=2)} m/s", True)
                    log("", f"Rot comp angle multiplier: {np.round(rotation_compensation_angle, decimals=2)} , forward multiplier: {np.round(rotation_compensation_1,decimals=2)}", True)
          


def setup_driver():

    WBP_controller = WheelBasePoseController(name="wheel_base_pose_controller",
                                                   open_loop_wheel_controller=
                                                        DifferentialController(name="diff_controller",
                                                                                wheel_radius=0.03, wheel_base=0.1125),
                                                    is_holonomic=False)

    if not SHOW_VEL_SPHERES:
        return
    for _ in range(num_robots):
        for _ in range(3):
            omni.kit.commands.execute(
                "CreateAndBindMdlMaterialFromLibrary",
                mdl_name="OmniPBR.mdl",
                mtl_name=f"OmniPBR",
                mtl_created_list=mtl_created_list,
            )

def setup_post_load_driver(robots):
    initialize_v_rho0_i(robots)