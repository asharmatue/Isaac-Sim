from omni.isaac.core.objects import FixedCuboid, VisualCuboid

import numpy as np
import json
import os

from Settings.settings import SITUATION_NAME, SITUATIONS_PATH
import Settings.util as util

def create_cube(world, path:str, name:str, id:int, position:np.ndarray, scale:np.ndarray, color:np.ndarray):
    world.scene.add(
        FixedCuboid(
            prim_path=f"/World/{path}/{name}{id:02}",
            name=f"{name.lower()}{id:02}",
            translation=position,
            scale=scale,  
            color=color,
        )
    )

def create_floor_cube(world, path:str, name:str, id:int, position:np.ndarray, scale:np.ndarray, color:np.ndarray):
    world.scene.add(
        VisualCuboid(
            prim_path=f"/World/{path}/{name}{id:02}",
            name=f"{name.lower()}{id:02}",
            translation=position,
            scale=scale,  
            color=color,
        )
    )

wall_index = 0
def add_wall(world, position:np.ndarray, scale:np.ndarray, color:np.ndarray):
    global wall_index
    create_cube(world, "Walls", "Cube", wall_index, position, scale, color)
    wall_index += 1

def load_situation():
    file_address = f"{os.path.dirname(__file__)}/{SITUATIONS_PATH}/{SITUATION_NAME}.json"
    if not os.path.isfile(file_address):
        util.debug_log("Situation", f"Could not load situation from file address {file_address}")
        return
    
    with open(file_address, 'r') as file:
        situation = json.load(file)
    
    return situation

def create_situation_walls(world, walls_color:np.ndarray):
    situation = load_situation()
    for situationJSON in situation["wall"]:
        position = situationJSON['position']
        position = np.array([position['x'], position['z'], position['y']])
        scale = situationJSON['scale']
        scale = np.array([scale['x'], scale['z'], scale['y']])
        add_wall(world, position, scale, walls_color)
