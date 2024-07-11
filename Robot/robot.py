from omni.isaac.wheeled_robots.robots import WheeledRobot
import numpy as np
from Perception.grid import get_grid_rho, get_xi_rho
from Settings.settings import actual_environment_x_min, actual_environment_y_min
from ExportSimulationData.data_values import addExportValue, ExportValue
from Perception.grid import normalized_x_steps, normalized_y_steps

import time

base_robot_name="robot_"

class Robot:
    index:int
    instance:WheeledRobot
    pos:np.ndarray
    ori:np.ndarray
    euler_ori:np.ndarray
    vel:np.ndarray
    rho:list[int]
    xi_rho:float
    p_rho_i:list[float]
    v_rho0:float
    
    def __init__(self, world, index:int):
        self.index = index
        self.instance = world.scene.get_object(f"{base_robot_name}{self.index:02}")
        self.update()
        self.__add_export_robot_data_values__()

    def update(self):
        # Compute pos, requires set instance
        pos, ori = self.instance.get_world_pose()
        self.pos, self.ori = np.array(pos), np.array(ori)

        # Compute vel, requires set instance
        vel = self.instance.get_linear_velocity()
        self.vel = np.array(vel)
        
        # Compute euler rotation, require up to date ori
        self.euler_ori = self.__compute_euler_ori__()

        # Compute rho, requires up to date pos
        self.rho = get_grid_rho(self.pos)

        # Compute xi rho, requires up to date rho
        self.xi_rho = get_xi_rho(self.rho[0], self.rho[1])

        # Compute p_rho_i, requires up to date rho
        self.p_rho_i = self.__compute_p_rho_i__()

    def __compute_euler_ori__(self):
        w, x, y, z = self.ori

        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = np.arctan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = np.arcsin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = np.arctan2(t3, t4)

        roll_x_nrad, pitch_y_nrad, yaw_z_nrad = [roll_x % 2*np.pi,  pitch_y % 2*np.pi, yaw_z % 2*np.pi]
        roll_x_deg, pitch_y_deg, yaw_z_deg = [np.rad2deg(roll_x),np.rad2deg(pitch_y),np.rad2deg(yaw_z)]
        roll_x_ndeg, pitch_y_ndeg, yaw_z_ndeg = [roll_x_deg % 360,  pitch_y_deg % 360, yaw_z_deg % 360]

        return np.array([roll_x, pitch_y, yaw_z]) # in radians -pi to pi
                
    def __compute_p_rho_i__(self):
        '''Compute p rho i, requires up to date rho'''
        robot_p_rho_x = ((actual_environment_x_min + self.rho[0]*normalized_x_steps) + 
                         (actual_environment_x_min + (self.rho[0]+1)*normalized_x_steps)) /2
        robot_p_rho_y = ((actual_environment_y_min + self.rho[1]*normalized_y_steps) + 
                         (actual_environment_y_min + (self.rho[1]+1)*normalized_y_steps)) /2

        # Center point (in positional meters) of the cell robot i is currently occupying
        return [robot_p_rho_x, robot_p_rho_y, 0]
    
    def __add_export_robot_data_values__(self):

        addExportValue(ExportValue(f"time", lambda : str(time.time())))

        # Add position export values for this robot
        addExportValue(ExportValue(f"robot-{self.index:02}-pos-x", lambda : str(self.pos[0])))
        addExportValue(ExportValue(f"robot-{self.index:02}-pos-y", lambda : str(self.pos[1])))
        addExportValue(ExportValue(f"robot-{self.index:02}-pos-z", lambda : str(self.pos[2])))

        # Add velocity export values for this robot
        addExportValue(ExportValue(f"robot-{self.index:02}-vel-x", lambda : str(self.vel[0])))
        addExportValue(ExportValue(f"robot-{self.index:02}-vel-y", lambda : str(self.vel[1])))
        addExportValue(ExportValue(f"robot-{self.index:02}-vel-z", lambda : str(self.vel[2])))

        