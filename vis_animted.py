
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import * 
from PyQt5.QtGui import * 
from PyQt5.QtCore import QThread
from PyQt5.QtCore import QSize
from src import stream
from PyQt5.QtCore import Qt
import vispy.scene
from vispy.scene import visuals
canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
view = canvas.central_widget.add_view()
view.camera = vispy.scene.cameras.TurntableCamera()

axis = visuals.XYZAxis(parent=view.scene)

grid = visuals.GridLines()
view.add(grid)

if __name__ == '__main__':
    import sys
    if sys.flags.interactive != 1:
         vispy.app.run()