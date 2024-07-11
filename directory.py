import os
import sys

def directory_setup():
    sys.path.append(str(os.path.dirname(__file__)))

# Setup the directory when imported
# directory_setup()