import time
import torch


def time_synchronized():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()
class Timer(object):
    def __init__(self):
        self.start_time=0
        self.end_time=0
        pass

        return time.time()
    def tic(self):
        self.start_time=time_synchronized()
    def toc(self):
        self.end_time=time_synchronized()
        
    
    
    @property
    def time_interval(self):
        '''
        This gives the execution time in seconds
        '''
        return self.end_time-self.start_time
    
    @property
    def fps(self):
        '''
        This gives the execution time in seconds
        '''
        return round(1./max((self.end_time-self.start_time),0.0000000000001))