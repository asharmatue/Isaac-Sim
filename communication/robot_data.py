from robot import Robot
import numpy as np
import random

class RobotData():
    serial:str
    position:dict[str, float]
    rotation:dict[str, float]
    def __init__(self, robot:Robot):
        self.serial = f"{robot.index:08}"
        x, z, y = robot.pos[0].item(), robot.pos[1].item(), robot.pos[2].item()
        self.position = {
            'x':x,
            'y':y,
            'z':z
        }
        roll, pitch, yaw = robot.euler_ori[0].item(), robot.euler_ori[1].item(), robot.euler_ori[2].item()
        self.rotation = {
            'pitch':np.rad2deg(pitch),
            'yaw':np.rad2deg(yaw),
            'roll':np.rad2deg(roll)
        }


class RandomRobot(Robot):
    index:int
    pos:np.ndarray
    vel:np.ndarray

    def __init__(self):
        self.index = random.randint(0, 2)
        self.pos = np.array([float(random.random() * 5) - 2.3, float(random.random() * 3.5) -1.1, float(0.1)])
        self.euler_ori = np.array([float(0), float(0), float(random.random() * 2 * np.pi - np.pi)])