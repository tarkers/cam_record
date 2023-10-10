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
        return round(1/(self.end_time-self.start_time))