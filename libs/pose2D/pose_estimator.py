from abc import ABC, abstractmethod
class Pose2Destimator(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def image_estimation(self, img_name):
        pass

