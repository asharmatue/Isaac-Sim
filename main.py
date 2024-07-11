from omni.isaac.examples.base_sample import BaseSample

import numpy as np
from omni.isaac.examples.user_examples.git_isaac_sim.directory import directory_setup
directory_setup()

import omni                                                     # Provides the core omniverse apis
from omni.isaac.range_sensor import _range_sensor               # Imports the python bindings to interact with lidar sensor

from omni.isaac.wheeled_robots.controllers.wheel_base_pose_controller import WheelBasePoseController
from omni.isaac.wheeled_robots.controllers.differential_controller import DifferentialController


stage = omni.usd.get_context().get_stage()                      # Used to access Geometry
timeline = omni.timeline.get_timeline_interface()               # Used to interact with simulation
lidarInterface = _range_sensor.acquire_lidar_sensor_interface() # Used to interact with the LIDAR

from settings import num_robots, forward_gain, angle_gain
from settings import show_log_send_robot_actions, EXPORT_DATA
import data_export
from environment_setup import setup_environment
from robot_setup import setup_robots
from util import log, performance_timestamp, mod
from robot import Robot
from communicator import send_robot_data

from Controller.robot_driver import setup_driver, setup_post_load_driver, apply_velocities

# For Obstacle Collision Point Visualisation Spheres in find_colliion_points()
obs_counter = 0
highest_obs_index = [0 for _ in range(num_robots)]


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
        setup_driver()
    
        
    async def setup_post_load(self):
        self._world = self.get_world()
        self.robots = [None for _ in range(num_robots)]
        for robot_index in range(num_robots):
            self.robots[robot_index] = Robot(self._world, robot_index)

        data_export.exportHeader() # After adding export values in Robot class.
            
        self._world.add_physics_callback("sending_actions", callback_fn=self.send_robot_actions)
        # Initialize our controller after load and the first reset
        
        self._WBP_controller = WheelBasePoseController(name="wheel_base_pose_controller",
                                                        open_loop_wheel_controller=
                                                            DifferentialController(name="diff_controller",
                                                                                    wheel_radius=0.03, wheel_base=0.1125),
                                                    is_holonomic=False)
        
        setup_post_load_driver(self.robots)
        return
    

    def send_robot_actions(self, step_size):
        for robot in self.robots:
            robot.update()
            send_robot_data(robot)

        data_export.exportData()

        apply_velocities(self.robots, self.get_world()) 
          
        return
     