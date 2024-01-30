from enum import Enum,IntEnum
class ExtendedEnum(Enum):
    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


class CAMERATYPE(ExtendedEnum):
    REALSENSE = "Realsense"
    AX700 = "AX700"


class VIDEOTYPE(ExtendedEnum):
    MP4 = "mp4"
    AVI = "avi"
    
class ERRORTYPE(ExtendedEnum):
    CAMERAERROR = 0
    VIDEOARROR = 1

class MEDIATYPE(ExtendedEnum):
    Video = 0
    Picture = 1
    Stream = 2
    IMAGES = 3
    


class DataType(Enum):
    DEFAULT = {"name": "default", "tips": "", "filter": ""}
    IMAGE = {"name": "image", "tips": "",
             "filter": "Image files (*.jpeg *.jpg *.png *.tiff *.psd *.pdf *.eps *.gif)"}
    VIDEO = {"name": "video", "tips": "",
             "filter": "Video files ( *.WEBM *.MPG *.MP2 *.MPEG *.MPE *.MPV *.OGG *.MP4 *.M4P *.M4V *.AVI *.WMV *.MOV *.QT *.FLV *.SWF)"}
    CSV = {"name": "csv", "tips": "",
           "filter": "Video files (*.csv)"}
    FOLDER = {"name": "folder", "tips": "", "filter": ""}

class MessageType(IntEnum):
    INFORMATION=1,
    QUESTION=2,
    WARNING=3,
    CRITICAL=4,

class MessageButtonType(IntEnum):
    YES=0,
    No=1,
    Cancel=2,
    OK=3,