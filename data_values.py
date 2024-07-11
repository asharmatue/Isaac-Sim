from collections.abc import Callable

class ExportValue:

    def __init__(self, header:str, value_reference:Callable[[], str]):
        self.header = header
        self.value_reference = value_reference

    def getHeaderString(self) -> str:
        return self.header

    def getExportString(self) -> str:
        return self.value_reference()
    
class ExportCompound(ExportValue):

    def __init__(self, *export_values:ExportValue):
        self.export_values = list(export_values)

    def getHeaderString(self) -> str:
        return ','.join(map(lambda h:h.getHeaderString(), self.export_values))

    def getExportString(self) -> str:
        return ','.join(map(lambda h:h.getExportString(), self.export_values))
    
    def addExportValue(self, value:ExportValue):
        self.export_values.append(value)

export_data :ExportCompound = ExportCompound()

def getExportData() -> ExportValue:
    '''Returns the export data that is currently set.'''
    return export_data

def addExportValue(value:ExportValue) -> ExportValue:
    '''Add the specified export value to the export data, returns the new export data.'''
    export_data.addExportValue(value)
    return export_data

def resetExportData() -> ExportValue:
    '''Resets the export data to an empty export compound, returns the empty export value.'''
    export_data = ExportCompound()
    return export_data

