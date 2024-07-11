from Settings.util import log, debug_log

class Header():
    version = 0
    '''Specifies the header version to check header data for.'''

    def __init__(self, header_data:bytes):
        '''Constructor for header using header data to extract values.'''

        self.message_size = 0
        '''Describes number of bytes in the message'''

        if (len(header_data) != Header.get_size()):
            debug_log(f"Header v{Header.version}", f"Header data has {len(header_data)}" +
                f" amount of bytes, while this version expects {Header.get_size()}")
            return
        
        if (header_data[0] != Header.version):
            debug_log(f"Header v{Header.version}", f"Header version {header_data[0]}" + 
                    f" does not match this header version {Header.version}")
            return
        
        self.message_size = int.from_bytes(header_data[1:5], 'big')
        debug_log(f"Header v{Header.version}", f"Header received message size {self.message_size}")

    def create(message_data:bytes) -> bytes:
        '''Get the header bytes'''
        header = bytes()

        # 1 byte header version data
        header += Header.version.to_bytes(1, byteorder="big", signed=False)
        
        # 4 bytes specifying message length
        header += len(message_data).to_bytes(4, byteorder="big", signed=False)

        debug_log(f"Header v{Header.version}", "Created header data:")
        header_data = header.hex("|")
        debug_log("", f"[{header_data}]")
        return header
    
    def get_size() -> int:
        '''Get the size of the header in number of bytes.'''
        return 5