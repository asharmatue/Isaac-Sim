from Settings.util import log, debug_log
from ServerCommunication.Connection.message import Message

from socket import socket, SHUT_RD, SHUT_WR
from queue import Queue 
from threading import Thread
import time
from collections.abc import Callable
from Settings.settings import DEBUG_LOG

class Connection():

    def __init__(self, connected_socket:socket, auto_start:bool=True):
        '''Create a connected with a connected socket.'''
        log("Connection", "Initializing connection for " +
            f"connected socket {connected_socket.getsockname()}")
        self.socket = connected_socket
        self._writing = True
        self._reading = True
        self._started = False
        self._closed = False

        #tcp_socket.settimeout(60)

        self._data_to_send = Queue[bytes]()

        self.receiver = Thread(target=self.__receiver__)
        self.sender = Thread(target=self.__sender__)

        def end_of_file_listener(called_socket:socket):
            if (called_socket == self.socket):
                self.__stop_reading__()
                self.__stop_writing__()
        self.end_listener = end_of_file_listener
        Message.add_end_of_file_listener(self.end_listener)

        if (auto_start): self.start()

    def start(self):
        '''Start the connection to begin reading and writing data'''
        if not self._started:
            debug_log("Connection", "Starting receiver and sender thread")
            self.receiver.start()
            self.sender.start()
            self._started = True

    def __stop_reading__(self):
        '''Stop the connection from reading on the socket'''
        if (self._reading):
            debug_log("Connection", "Stop reading from the socket")
            self._reading = False
            self.socket.shutdown(SHUT_RD)
    
    def __stop_writing__(self):
        '''Stop the connection from writing on the socket'''
        if (self._writing):
            debug_log("Connection", "Stop writing to the socket")
            self._writing = False
            self.send_message("") # Unblock sender from waiting
            self.socket.shutdown(SHUT_WR)
        
    def close(self):
        '''Close the connection of the socket to the current server'''
        if (self._closed):
            debug_log("Connection", "Connection has already been closed")
            return
        
        log("Connection", f"Closing the TCP connection with socket {self.socket.getsockname()}")
        
        self.__stop_writing__()
        debug_log("Connection", "Waiting for sender thread to join")
        self.sender.join()
        
        debug_log("Connection", "Waiting for reading to stop " +
                                 "from EOF signal or 4 second timeout")
        self.socket.settimeout(4)
        while(self._reading):
            time.sleep(0.1)
        
        debug_log("Connection", "Waiting for receiver thread to join")
        self.receiver.join()
            
        debug_log("Connection", f"Closing connection at {self.socket.getsockname()} socket to address {self.socket.getpeername()}")
        self.socket.close()
        Message.remove_end_of_file_listener(self.end_listener)
        self._closed = True

    def send_message(self, data:bytes):
        '''Add a message to the connection to send'''
        debug_log("Connection", "Adding message to the queue:")
        debug_log("", f"\'{data}\'")
        self._data_to_send.put(data)

    _receive_listeners =  list[Callable[[bytes], None]]()
    '''List of listeners for receiving connection data'''

    def add_receive_listener(self, listener: Callable[[bytes], None]):
        '''Add listener for receiving connection data'''
        self._receive_listeners.append(listener)

    def __receiver__(self):
        '''Program to run on a thread, which continously receives messages from the connection'''
        debug_log("Receiver", "Start of receiver thread")

        while self._reading:
            received_data = self.__receive__()
            debug_log("Receiver", "Received message from the connection")
        
            if (not received_data): 
                continue
        
            for listener in self._receive_listeners:
                listener(received_data)

        debug_log("Receiver", "Reached end of receiver thread")

    def __sender__(self):
        '''Program to run on a thread, which writes all messages to the connection'''
        debug_log("Sender", "Start of sender thread")

        while True:
            message = self._data_to_send.get()
            debug_log("Sender", "Got new data to send from queue")

            if (not self._writing): 
                break # End sender thread as we should stop writing 
            if (DEBUG_LOG):
                log("Sender", "Sending message:")
                log("", f"{message}")
            self.__send__(message)
        
        debug_log("Sender", "Reached end of sender thread")

    def __receive__(self) -> bytes:
        '''Wait to receive a message from the socket connection'''
        try:
            debug_log("Connection", "Waiting to receive message")
            return Message.receive(self.socket)
        except TimeoutError:
            debug_log("Connection", "TimeOutError trying to receive message " +
                      f"at socket {self.socket.getsockname()}")
            self.__stop_reading__()
        except Exception as exception:
            debug_log("Connection", "Exception occurred trying to receive message " +
                      f"at socket {self.socket.getsockname()}")
            debug_log("Exception", f"{exception}", True)
            self.__stop_reading__()

    def __send__(self, data:bytes):
        '''Send a message to the socket connection'''
        try:
            debug_log("Connection", "Sending message")
            Message(data).send(self.socket)
        except TimeoutError:
            debug_log("Connection", "TimeOutError trying to send message " +
                      f"at socket {self.socket.getsockname()}")
            self.__stop_writing__()
        except Exception as exception:
            debug_log("Connection", "Exception occurred trying to send message " +
                      f"at socket {self.socket.getsockname()}")
            debug_log("Exception", f"{exception}", True)
            self.__stop_writing__()
