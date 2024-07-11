from data_values import export_data
from settings import EXPORT_FILE_ADDRESS
import os

def getExportFileAddress():
   file_address = f"{os.path.dirname(__file__)}/{EXPORT_FILE_ADDRESS}.csv"
   return file_address

def exportString(fileAddress:str, string:str):
    '''Append a specified string to the specified file.'''
    with open(fileAddress, "a") as outfile:
        outfile.write(string)

def exportValue(value:str):
    '''Export a specified string value to the export file.'''
    exportString(getExportFileAddress(), value + "\n")

def exportHeader():
    '''Write the header string for the current export data to the export file.'''
    exportValue(export_data.getHeaderString())

def exportData():
    '''Write the current export data string to the export file.'''
    exportValue(export_data.getExportString())

def clearData():
    '''Clear the export file from all previous data.'''
    with open(getExportFileAddress(), "w") as outfile:
        outfile.write("")