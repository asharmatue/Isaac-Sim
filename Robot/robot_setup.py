from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.wheeled_robots.robots import WheeledRobot
from omni.isaac.core.objects import VisualSphere

import numpy as np
from Perception.lidar import create_lidars
from Settings.settings import num_robots 
from Settings.settings import lineMode, typeRobot, lidarsDrawLines, lidarsDrawPoints, SHOW_VEL_SPHERES

def create_robots(world, typeRobot, lineMode):
    assets_root_path = get_assets_root_path()
    jetbot_asset_path = assets_root_path + "/Isaac/Robots/Jetbot/jetbot.usd"
    kaya_asset_path = assets_root_path + "/Isaac/Robots/Kaya/kaya.usd"
    base_robot_prim_path = "/World/Robot_"
    base_robot_name="robot_"
    spawn_orientation = np.array([1,0,0,0])

    if lineMode:
        robot_spawn_positions_x_limits = [-0.7 , 0.7]
        robot_spawn_positions_y_limits = [0, 0]
        robot_spawn_positions_x = np.linspace(robot_spawn_positions_x_limits[0], robot_spawn_positions_x_limits[1], num_robots)
        robot_spawn_positions_y = np.linspace(robot_spawn_positions_y_limits[0], robot_spawn_positions_y_limits[1], num_robots)

        for robot_index in range(num_robots):
                if typeRobot == 1:       # Jetbot
                    world.scene.add(
                    WheeledRobot(
                        prim_path=f"{base_robot_prim_path}{robot_index:02}",
                        name=f"{base_robot_name}{robot_index:02}",
                        wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
                        create_robot=True,
                        usd_path=jetbot_asset_path,
                        position=[robot_spawn_positions_x[robot_index],robot_spawn_positions_y[robot_index],0],
                        orientation=spawn_orientation,
                    )
                )
                elif typeRobot ==  2:    # Kaya
                    world.scene.add(
                    WheeledRobot(
                        prim_path=f"{base_robot_prim_path}{robot_index:02}",
                        name=f"{base_robot_name}{robot_index:02}",
                        wheel_dof_names=["axle_0_joint", "axle_1_joint","axle_2_joint"],
                        create_robot=True,
                        usd_path=kaya_asset_path,
                        position=[robot_spawn_positions_x[robot_index],robot_spawn_positions_y[robot_index],0],
                        orientation=spawn_orientation,
                    )
                )    
                        
                    
    else:
        robot_spawn_positions_x_limits = [-0.7 , 0.7] 
        robot_spawn_positions_y_limits = [-0.7 , 0.4]
        robot_cols = int(np.ceil(np.sqrt(num_robots)))
        robot_rows = int(np.ceil(num_robots / robot_cols))
        robot_spawn_positions_x = np.linspace(robot_spawn_positions_x_limits[0], robot_spawn_positions_x_limits[1], robot_cols)
        robot_spawn_positions_y = np.linspace(robot_spawn_positions_y_limits[0], robot_spawn_positions_y_limits[1], robot_rows)

        robot_index = 0
        for y in range(robot_rows):
            for x in range(robot_cols):
                if robot_index < num_robots:
                    if typeRobot == 1:       # Jetbot
                        world.scene.add(
                            WheeledRobot(
                                prim_path=f"{base_robot_prim_path}{robot_index:02}",
                                name=f"{base_robot_name}{robot_index:02}",
                                wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
                                create_robot=True,
                                usd_path=jetbot_asset_path,
                                position=[robot_spawn_positions_x[x % robot_cols],robot_spawn_positions_y[y % robot_rows],0],
                                orientation=spawn_orientation,
                            )
                        )
                    elif typeRobot ==  2:    # Kaya
                        world.scene.add(
                            WheeledRobot(
                                prim_path=f"{base_robot_prim_path}{robot_index:02}",
                                name=f"{base_robot_name}{robot_index:02}",
                                wheel_dof_names=["axle_0_joint", "axle_1_joint","axle_2_joint"],
                                create_robot=True,
                                usd_path=kaya_asset_path,
                                position=[robot_spawn_positions_x[x % robot_cols],robot_spawn_positions_y[y % robot_rows],0],
                                orientation=spawn_orientation,
                            )
                        )
                    robot_index += 1
    return



# Base paths needed for create_vel_vis() and in velocity_commands() to get color values
base_sphere_prim_path = "/World/Robot_"

base_entering_sphere_prim_path_suffix = "/Entering_Sphere_"
base_exploration_sphere_prim_path_suffix = "/Exploration_Sphere_"
base_interaction_sphere_prim_path_suffix = "/Interaction_Sphere_"

base_sphere_prim_path_suffix = []
base_sphere_prim_path_suffix.append(base_entering_sphere_prim_path_suffix)
base_sphere_prim_path_suffix.append(base_exploration_sphere_prim_path_suffix)
base_sphere_prim_path_suffix.append(base_interaction_sphere_prim_path_suffix)

def create_vel_vis(world):

    base_entering_sphere_name = "entering_sphere_"
    base_exploration_sphere_name = "exploration_sphere_"
    base_interaction_sphere_name = "interaction_sphere_"

    base_sphere_name = []
    base_sphere_name.append(base_entering_sphere_name)
    base_sphere_name.append(base_exploration_sphere_name)
    base_sphere_name.append(base_interaction_sphere_name)

    entering_trans = -0.05
    exploration_trans = 0.0
    interaction_trans = 0.05 

    vel_trans = []
    vel_trans.append(entering_trans)
    vel_trans.append(exploration_trans)
    vel_trans.append(interaction_trans)

    entering_color = np.array([2.0, 0.0, 0.0])
    exploration_color = np.array([0.0, 2.0, 0.0])
    interaction_color = np.array([0.0, 0.0, 2.0])

    vel_color = []
    vel_color.append(entering_color)
    vel_color.append(exploration_color)
    vel_color.append(interaction_color)

    for i in range(num_robots):
        for v in range(3):
            world.scene.add(
                    VisualSphere(
                        prim_path=f"{base_sphere_prim_path}{i:02}/chassis{base_sphere_prim_path_suffix[v]}{i:02}",
                        name=f"{base_sphere_name[v]}{i:02}",
                        translation=np.array([vel_trans[v], 0, 0.15]),
                        scale=np.array([0.02, 0.02, 0.02]),  
                        color=vel_color[v]
                    ))


def setup_robots(world):
    create_robots(world, typeRobot, lineMode)           # Create Robots
    create_lidars(lidarsDrawLines, lidarsDrawPoints)    # Create Lidars
    if SHOW_VEL_SPHERES:
        create_vel_vis(world)                           # Create Velocity Visualisation Spheres
    
    return

