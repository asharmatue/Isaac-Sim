from Settings.util import log, debug_log
from ServerCommunication.Connection.TCP_client import create_tcp_connection
from Settings.settings import CREATE_CONNECTION, DEBUG_LOG


class Channel():
    def __init__(self):
        self.connection = create_tcp_connection()
        self.connection.add_receive_listener(Channel.on_receive_message)

    def send_message(self, message:str):
        if (DEBUG_LOG):
            log("Channel", "Sending message:")
            log("", f"{message}")
        self.connection.send_message(message.encode("utf-8"))

    def on_receive_message(data:bytes):
        message = data.decode("utf-8")
        if (DEBUG_LOG):
            log("Channel", "Received message")
            log("", f"\'{message}\'")

    def close(self):
        self.connection.close()

    def is_finished(self):
        return not self.connection._reading and not self.connection._writing

channel = None

try:
    if CREATE_CONNECTION:
        channel = Channel()
except (Exception):
    debug_log("TCP Client", "Exception occurred")

def send_message(message:str):
    if channel: 
        channel.send_message(message)