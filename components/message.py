from PyQt5 import QtWidgets
import sys
from util.utils import MessageType,MessageButtonType
class MessageBox(QtWidgets.QMessageBox):
    def __init__(self,parent=None):
        super(MessageBox, self).__init__(parent)
        pass

    def show(self,form,data,type=MessageType.WARNING):

        if type ==MessageType.WARNING:
            self.setText(data)
            self.setIcon(type)   
            self.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No | QtWidgets.QMessageBox.Cancel)   
        elif type ==MessageType.CRITICAL:
            self.setText(data)
            self.setIcon(type)   
            self.setStandardButtons(QtWidgets.QMessageBox.Ok)   

            
        # if type==MessageType.INFORMATION:
        #     return self.information(form, 'information',  data)
        # elif type == MessageType.QUESTION:
        #     return self.question(form, 'question',  data)
        # elif type == MessageType.WARNING:
        #    return  self.warning(form, 'warning',  data)
        # elif type == MessageType.CRITICAL:
        #    return self.critical(form, 'critical', data)

        ret = self.exec()  
        if ret == QtWidgets.QMessageBox.Yes:
            return MessageButtonType.YES
        elif ret == QtWidgets.QMessageBox.No:
            return MessageButtonType.No
        elif ret == QtWidgets.QMessageBox.Cancel:
            return MessageButtonType.Cancel
        elif ret == QtWidgets.QMessageBox.Ok:
            return MessageButtonType.OK

