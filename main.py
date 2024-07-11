from omni.isaac.examples.base_sample import BaseSample
from Settings.settings import num_robots, EXPORT_DATA
from Environment.environment_setup import setup_environment
from Robot.robot_setup import setup_robots
from Robot.robot import Robot
from Controller.robot_driver import setup_driver, setup_post_load_driver, apply_velocities
from ServerCommunication.communicator import send_robot_data
from ExportSimulationData.data_export import clearData, exportHeader, exportData

class Main(BaseSample):

    robots:list[Robot]
    '''List of robots in the simulation world'''

    def __init__(self) -> None:
        super().__init__()
        if (EXPORT_DATA):
            clearData()
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

        self._world.add_physics_callback("sending_actions", callback_fn=self.send_robot_actions)
        setup_post_load_driver(self.robots)
        exportHeader() # After adding export values in Robot class.
        return

    def send_robot_actions(self, step_size):
        for robot in self.robots:
            robot.update()
            send_robot_data(robot)
        apply_velocities(self.robots, self.get_world()) 
        exportData()
        return
     