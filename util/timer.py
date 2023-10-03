import time


class Timer(object):
    def __init__(self):
        self.start_time=0
        self.end_time=0
        pass
    
    
    def tic(self):
        self.start_time=time.time()
        
        
    def toc(self):
        self.end_time=time.time()
        pass
    
    def get_time_interval(self):
        return self.end_time-self.start_time