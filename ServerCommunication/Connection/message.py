from Settings.util import debug_log
from ServerCommunication.Connection.header import Header

from socket import socket
from collections.abc import Callable

class Message():

    def __init__(self, data:bytes):
        '''Create a message with specified data payload'''
        self.data = data

    def send(self, socket:socket):
        '''Send the created message to the specified socket'''
        debug_log("Message", "Sending message header")
        socket.send(Header.create(self.data))

        debug_log("Message", "Sending message data")
        socket.send(self.data)

    def receive(socket:socket) -> bytes:
        '''Wait to receive a message from the specified socket'''
        header_data = Message.receive_sized_message(socket, Header.get_size())

        if (not header_data): 
            return

        debug_log("Message", "Received message header")
        message_data = Message.receive_sized_message(socket, Header(header_data).message_size)

        if (not message_data): 
            return

        debug_log("Message", "Received message data")
        return message_data
    
    def receive_sized_message(socket:socket, size:int) -> bytes:
        '''Receive a message of specified size from the socket'''
        data_received = bytearray()

        # Continue to read data until we got the specified amount
        while len(data_received) < size:
            data = socket.recv(size)

            # Check if data was received
            if not data:
                debug_log("Message", "Received End Of File signal")
                # No data received is End Of File signal, so we alert listeners
                Message.end_of_file_handler(socket)
                return
            
            # Add data to currently received data buffer
            data_received.extend(data)

        return bytes(data_received)
    
    __end_of_file_listeners__ =  list[Callable[[socket], None]]()
    '''List of listeners for the End Of File signal'''

    def add_end_of_file_listener(listener:Callable[[socket], None]):
        '''Add a listener to the End Of File signal'''
        debug_log("Message", "Adding End Of File listener")
        Message.__end_of_file_listeners__.append(listener)

    def remove_end_of_file_listener(listener:Callable[[socket], None]):
        '''Remove a listener for the End Of File signal'''
        debug_log("Message", "Removing End Of File listener")
        Message.__end_of_file_listeners__.remove(listener)

    def end_of_file_handler(socket:socket):
        '''Alert all the listeners of the End Of File signal'''
        for listener in Message.__end_of_file_listeners__:
            listener(socket)