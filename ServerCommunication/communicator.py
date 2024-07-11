import os, sys
# Setup the directory when imported
sys.path.append(str(os.path.dirname(__file__)))

from Settings.util import log, debug_log, add_debug_context
from ServerCommunication.Connection.robot_data import RobotData, RandomRobot
from Robot.robot import Robot
import time
from ServerCommunication.Timelines.timeline import save_robot_data
from ServerCommunication.Connection.channel import send_message, channel
from Settings.settings import SEND_TIMELINE
from ServerCommunication.Timelines.timeline import send_timeline, stop_timeline

import json

def send_robot_data(robot:Robot):
    """Sending Robot Data"""
    data = RobotData(robot)
    message = json.dumps(data.__dict__)
    save_robot_data(data)
    send_message(message)

if (__name__ == "__main__"):
    if SEND_TIMELINE:
        send_timeline()

    while True:
        log("TCP Client", "Waiting for message or type 'stop' to end connection:")
        input_string = input()
        if (input_string == "stop"): 
            break
        if (input_string == "debug"):
            add_debug_context("TCP Client")
            continue
        if (channel and channel.is_finished()):
            channel.close()
            break
        if (input_string == "robot"):
            send_robot_data(RandomRobot())
            continue
        if (input_string == "timeline"):
            send_timeline()
            continue
        send_message(input_string)

    if SEND_TIMELINE:
        stop_timeline()

    debug_log("TCP Client", "Closing connection")
    if channel:
        channel.close()