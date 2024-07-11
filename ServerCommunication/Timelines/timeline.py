from Settings.settings import SITUATION_NAME, SAVE_TIMELINE, SEND_TIMELINE, LIMIT_TIMELINE_RATE, TIMELINE_RATE, TIMELINE_PATH
from Settings.util import debug_log, log
from ServerCommunication.Connection.robot_data import RobotData
from ServerCommunication.Connection.channel import send_message

from threading import Thread
import time
import json
import os


last_save = time.time()
last_robot_save = dict()
timeline_active = False

def get_timeline_file_address():
    file_address = f"{os.path.dirname(__file__)}/{TIMELINE_PATH}/{SITUATION_NAME}.json"
    return file_address

def save_robot_data(robot_data:RobotData):
    if not SAVE_TIMELINE:
        return
    debug_log("Timeline", "Trying to save robot data to timeline")
    global last_save
    waitTime = time.time() - last_save

    if robot_data.serial in last_robot_save:
        robotTimeDifference = time.time() - last_robot_save.get(robot_data.serial)
        if LIMIT_TIMELINE_RATE and TIMELINE_RATE > robotTimeDifference:
            return
    
    file_address = get_timeline_file_address()
    debug_log("Timeline", f"Adding robot data to timeline with file address {file_address}")

    with open(file_address, 'a') as file:
        file.write(f"{waitTime}\n")
        file.write(f"{json.dumps(robot_data.__dict__)}\n")
        last_save = time.time()
        last_robot_save[robot_data.serial] = last_save


def clear_timeline():
    file_address = get_timeline_file_address()
    debug_log("Timeline", f"Clearing timeline file with file address {file_address}")
    os.makedirs(os.path.dirname(file_address), exist_ok=True)
    with open(file_address, 'w') as file:
        file.write("")


last_load = time.time()

def timeline_thread():
    file_address = get_timeline_file_address()
    
    if not os.path.isfile(file_address):
        debug_log("Timeline", f"Could not load timeline from file address {file_address}")
        return
    
    log("Timeline", "Starting to send timeline data")
    global last_load
    
    with open(file_address, 'r') as file:
        global timeline_active
        while timeline_active:
            # Check if there is still a line to read.
            readline = file.readline()
            if not readline:
                timeline_active = False
                break

            waitTime = float(readline)
            debug_log("Timeline", f"Read wait time: {waitTime}")
            data = file.readline()
            debug_log("Timeline", f"Read timeline data: {data}")

            processTime = time.time() - last_load

            debug_log("Timeline", f"Waiting: {waitTime - processTime}s")
            if (waitTime - processTime > 0):
                time.sleep(waitTime - processTime)

            last_load = time.time()

            debug_log("Timeline", f"Sending timeline data not load timeline from file address {file_address}")
            send_message(data)

    log("Timeline", "Finished sending timeline data")

thread = Thread(target=timeline_thread)
def send_timeline():
    global timeline_active
    if (not timeline_active):
        debug_log("Timeline", "Sending data from timeline")
        timeline_active = True
        global thread
        thread = Thread(target=timeline_thread)
        thread.start()

def stop_timeline():
    global timeline_active
    if (thread.is_alive()):
        timeline_active = False
        thread.join()
    timeline_active = False

if SAVE_TIMELINE:
    clear_timeline()