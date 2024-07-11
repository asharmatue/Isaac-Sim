from omni.isaac.core.objects import FixedCuboid

from omni.isaac.core.objects.ground_plane import GroundPlane
import omni.isaac.core.utils.prims as prim_utils

from omni.isaac.sensor import Camera

import numpy as np

from Settings.settings import show_walls, show_door, show_grid_vis, show_old_grid_vis, show_victim_cube, show_test_wall
from Perception.grid import create_grid_vis, create_old_grid_vis
from Environment.environment import create_situation_walls

def setup_environment(world):
    ### Ground planes ###
    # world.scene.add_default_ground_plane()                                    # Default
    floor_color_toggle = 2
    if floor_color_toggle == 1:
        floor_color = np.array([0.1, 0.1, 0.4])         # Blue
    elif floor_color_toggle == 2:
        floor_color = np.array([1.0, 1.0, 1.0])         # White
    GroundPlane(prim_path="/World/groundPlane", size=10, color=floor_color)     # Custom
                                                        
    ### Lights ###
    light_1 = prim_utils.create_prim(
        "/World/Light_1",
        "SphereLight",
        position=np.array([0.0, 0.0, 2.5]),
        attributes={
            "inputs:radius": 0.25,
            "inputs:intensity": 30e3,
            "inputs:color": (1.0, 1.0, 1.0)
        }
    )
    light_2 = prim_utils.create_prim(
        "/World/Light_2",
        "SphereLight",
        position=np.array([0.0, 1.25, 2.5]),
        attributes={
            "inputs:radius": 0.25,
            "inputs:intensity": 30e3,
            "inputs:color": (1.0, 1.0, 1.0)
        }
    )

    ### Camera ###
    TopDownCamera = Camera(
        prim_path="/World/TopDownCamera",
        position=np.array([0.0, 1.25, 13.0]),
        frequency=20,
        resolution=(256, 256),
        orientation=[0.0, 0.70711, 0.0, -0.70711]
        # [0, 0, 90] is [0.0, 0.70711, 0.0, -0.70711] || OR || rot_utils.euler_angles_to_quats(np.array([180, -90, 0]), degrees=True, extrinsic=False)  
    
    )
    TopDownCameraCenter = Camera(
        prim_path="/World/TopDownCameraCenter",
        position=np.array([0.0, 1.0, 12.8]),
        frequency=20,
        resolution=(256, 256),
        orientation=[0.0, 0.70711, 0.0, -0.70711]
    )

    ### Togglable Enviroment ###
    if show_walls:
        walls_color = np.array([1, 0.5, 0.5])
        # walls_visual_material = 
        # walls_physics_material = 
        #create_walls(world, walls_color)    # Create Walls
        create_situation_walls(world, walls_color) # Create walls from situation data


    if show_grid_vis:
        create_grid_vis(world)          # Create Greyscale Grid Visualisation
    if show_old_grid_vis:
        create_old_grid_vis(world)      # Create Grid Cell Visualisation
    if show_victim_cube:
        victim_cube = world.scene.add(
            FixedCuboid(
                prim_path="/World/Victim_Cube",
                name="victim_cube",
                translation=np.array([-1.3, 1.6, 0.03]),
                scale=np.array([0.05, 0.05, 0.05]),  
                color=np.array([0.0, 1.0, 0.0]),
                # visual_material=walls_visual_material,
                # physics_material=walls_physics_material,
            )
        )
    if show_door:
        # Door
        Cube_10 = world.scene.add(
            FixedCuboid(
                prim_path="/World/Walls/Cube_10",
                name="cube_10",
                translation=np.array([-0.625, 0.9, 0.25]),
                scale=np.array([0.45, 0.1, 0.5]),  
                color=np.array([0.5, 0, 0]),
                # visual_material=walls_visual_material,
                # physics_material=walls_physics_material,
            )
        )
    if show_test_wall:
        # Test Wall for find_collision_points()
        Cube_11 = world.scene.add(
            FixedCuboid(
                prim_path="/World/Walls/Cube_11",
                name="cube_11",
                translation=np.array([-1.4, 0.0, 0.25]),
                scale=np.array([0.1, 0.45, 0.5]),  
                color=np.array([0.5, 0, 0]),
                # visual_material=walls_visual_material,
                # physics_material=walls_physics_material,
            )
        )