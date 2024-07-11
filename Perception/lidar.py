import numpy as np
import omni
from settings import num_robots

from omni.isaac.range_sensor import _range_sensor               # Imports the python bindings to interact with lidar sensor 

timeline = omni.timeline.get_timeline_interface()               # Used to interact with simulation
lidarInterface = _range_sensor.acquire_lidar_sensor_interface() # Used to interact with the LIDAR


def create_lidars(lidarsDrawLines, lidarsDrawPoints):
    base_lidar_path = "/Lidar_"
    base_lidar_parent = "/World/Robot_"
    for i in range(num_robots):
        omni.kit.commands.execute(
            "RangeSensorCreateLidar",
            path=f"{base_lidar_path}{i:02}",
            parent=f"{base_lidar_parent}{i:02}/chassis",
            min_range=0.1,
            max_range=15.0,
            draw_points=lidarsDrawPoints,
            draw_lines=lidarsDrawLines,
            horizontal_fov=360.0,
            vertical_fov=30.0,
            horizontal_resolution=0.4,
            vertical_resolution=4.0,
            rotation_rate=0.0,
            high_lod=False,
            yaw_offset=0.0,
            enable_semantics=False
        )

    return

def get_lidar(robot_index):
        base_lidar_path = "/Lidar_"
        base_lidar_parent = "/World/Robot_"

        depth = lidarInterface.get_linear_depth_data(f"{base_lidar_parent}{robot_index:02}/chassis{base_lidar_path}{robot_index:02}")
        azimuth = lidarInterface.get_azimuth_data(f"{base_lidar_parent}{robot_index:02}/chassis{base_lidar_path}{robot_index:02}")

        return np.array(depth), np.array(azimuth)
