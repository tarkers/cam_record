from PyQt5 import QtWidgets
import sys
from util.enumType import MessageType,MessageButtonType
class MessageBox(QtWidgets.QMessageBox):
    def __init__(cls,parent=None):
        super(MessageBox, cls).__init__(parent)
        pass
    @classmethod    
    def show(cls,data,type=MessageType.WARNING):

        if type ==MessageType.WARNING:
            cls.setText(data)
            cls.setIcon(type)   
            cls.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No | QtWidgets.QMessageBox.Cancel)   
        elif type ==MessageType.CRITICAL:
            cls.setText(data)
            cls.setIcon(type)   
            cls.setStandardButtons(QtWidgets.QMessageBox.Ok)   

            
        # if type==MessageType.INFORMATION:
        #     return cls.information(form, 'information',  data)
        # elif type == MessageType.QUESTION:
        #     return cls.question(form, 'question',  data)
        # elif type == MessageType.WARNING:
        #    return  cls.warning(form, 'warning',  data)
        # elif type == MessageType.CRITICAL:
        #    return cls.critical(form, 'critical', data)

        ret = cls.exec()  
        if ret == QtWidgets.QMessageBox.Yes:
            return MessageButtonType.YES
        elif ret == QtWidgets.QMessageBox.No:
            return MessageButtonType.No
        elif ret == QtWidgets.QMessageBox.Cancel:
            return MessageButtonType.Cancel
        elif ret == QtWidgets.QMessageBox.Ok:
            return MessageButtonType.OK

