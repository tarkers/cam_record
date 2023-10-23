import math
from typing import List
from  numpy.linalg import norm
import numpy as np
import cv2
import array


def calculate_length(p1, p2):
    vector = p2 - p1
    return math.ceil(math.sqrt(float(vector[0] ** 2 + vector[1] ** 2)))


def point_to_line(p1, p2, p3):
    return norm(np.cross(p2 - p1, p1 - p3)) / norm(p2 - p1)


id_track = 0


class NodeCandidate(object):

    def __init__(self, node:list, frame_id:int, parent_node:'NodeCandidate'=None , 
        image_crop:'np.ndarray'=None,         
        ChildNode:'NodeCandidate'=None):
        global id_track
        '''
        node:[x,y,size]
        '''
        self.parent_node = parent_node
        self.child_node = ChildNode
        
        
        self._point = np.array(node[:2])
        self.is_on_path=False
        self.frame_id = frame_id
        self.size = node[2]
        self.rank = 0 
        self.image_crop=image_crop
        self._id = id_track
        id_track += 1
        
    @property
    def point(self):
        return self._point
    
    @property
    def ID(self):
        return self._id
    
    @property
    def vector(self):
        if self.child_node is None:
            return 0, 0
        p1 = self.point
        p2 = self.child_node.point
        direction = p2 - p1
        distance = calculate_length(p1, p2)
        return direction, distance
    
    def __str__(self):
        direction, distance = self.vector
        child_text = ""
        if self.child_node is not None:
            child_text = f"->{self.child_node.ID}".ljust(5) + \
            f"n:{self.child_node.point}".ljust(15) + \
            f"z:{self.child_node.size}".ljust(10) + \
            f"D:{distance}".ljust(10) + \
            f"v:{direction}".ljust(15)
                          
        return f"->{self.ID}".ljust(5) + \
            f"s:{self.point}".ljust(15) + \
            f"z:{self.size}".ljust(10) + \
            child_text + \
            f"rank:{self.rank}"


class PathCandidate(object):
    '''
    add all start path node
    '''
    def __init__(self):
        self.node_pool = []
     
    def add_node(self, node):
        self.node_pool.append(node)
        
    def remove_node(self, node):
        self.node_pool.remove(node)
    
    
class TrackBall(object):

    def __init__(self, fps=60):
        self.ball_candidate = []
        self.track_range = {}
        self.extract_path = []
        self.speed_minthreshold = 0
        self.speed_maxthreshold = 10
        self.node_candidate = []  # x,y,size,frame_idx
        self.new_node_candidate = []
        self.path_candidate = PathCandidate()
        self.node_dict = {}
        self.found_center = []
        self.ball_found = False
        self.out_off_bound = False
        self.fake_candidate = []
        self.path_line=[]

    
    def find_lost_path(self, node_list:List[NodeCandidate]  , n_f):
        '''
        check if ball candidate can be select
        '''
        s_length = len(self.new_node_candidate)
        
        for start_idx in range(s_length):
            start_node = self.new_node_candidate[start_idx]
            s_f = start_node.frame_id
            
            if start_node.child_node is not None :  # means it's on path
                continue
            
            for next_node in node_list:  # check if can generate tracklet
                distance = calculate_length(start_node.point, next_node.point) 
                if distance < 1:  # not track of static object
                    if start_node not in self.delete_node:
                        self.delete_node.append(start_node)
                    continue
                if distance > 20 * min((n_f - s_f) // 2,1):  # distance too far , start node may not be real track
                    if next_node not in self.new_node_candidate:
                        self.new_node_candidate.append(next_node)  # next_node may be root ball node
                    continue
                
                if (n_f - s_f) > 4:  # # remove lost track data 
                    if start_node not in self.delete_node:
                        print(f"will remove lost track: {start_node.ID}")
                        self.delete_node.append(start_node)
                    continue 
                
                if distance > 1:
                    if next_node.parent_node is None and start_node.child_node is None:
                        next_node.rank = start_node.rank + 1
                        start_node.child_node = next_node
                        next_node.parent_node = start_node
                        if start_node.rank > 2:
                            print(f"found ball path:{start_node.ID} {start_node.point}")
                            self.ball_found = True
                            ### test path ###
                            tnp = next_node
                            # extract_path = []
                            while tnp.parent_node is not None:
                                tnp.is_in_path=True
                                self.extract_path.append(tnp)
                                self.path_line.append(tnp.point)
                                tnp = tnp.parent_node
                            self.path_candidate.add_node(tnp)   #add the path start_node

                    elif next_node.parent_node is not None:
                        print("need to filter parent node")  # need to filter parent node
                    else:
                        print("need to filter child node")
            if self.ball_found:
                break            
            
        # remove static node
        for node in self.delete_node:
            self.new_node_candidate.remove(node)
            print(f"{node.ID}->node_remove")
            if node.child_node is not None:
                child_node = node.child_node
                node.child_node = None
                self.new_node_candidate.append(child_node)  # become root node
                tmp = node
                print(f"{child_node.ID} is new root")
                while tmp.child_node is not None:
                    tmp.child_node.rank -= 1
                    tmp = tmp.child_node
        if self.ball_found:
            print("found ball remove node-candidate")
            self.new_node_candidate=[]
            
        print(f"ball candidate: {len(self.new_node_candidate)}")    
         

    def track_path(self,node_list, frame_idx):
        '''
        when we found ball_candidates we start to narror serch area
        '''
        now_start=self.extract_path[0]
        found_next=False
        test=[]
        ## track by ball
        for new_candidate in node_list:
            distance = calculate_length(now_start.point, new_candidate.point) 
            
            if distance >0 and distance < 15 * min((frame_idx-now_start.frame_id ),1):
                self.extract_path.insert(0,new_candidate)
                self.path_line=[new_candidate.point] + self.path_line
                found_next=True
                break
        
        
        if not found_next:
            self.ball_found=False
            print("lost ball tracking")
            self.new_node_candidate.append(self.extract_path[0])    # append node back to candidate

        ## track by image
        
        return self.extract_path                
    
    def match_pic(self):
        pass

    def crop_image(self,img,node):
        x,y=node.point
        h,w,c=img.shape
        crop =img[max(y-50,0):min(h,y+50),max(x-50,0):min(w,x+50)]
        return crop
    
    def match_keypoints(self, ball_candidates, frame_idx,img):
        if len(self.new_node_candidate) == 0:  # first input set candidate inside
            for node in ball_candidates:
                self.node_dict[str(node + [frame_idx])] = [frame_idx, 0]
                
                tmp = NodeCandidate(node, frame_idx)
                crop=self.crop_image(img,tmp)
                tmp.image_crop=crop
                self.new_node_candidate.append(tmp)
            return self.extract_path 
        self.delete_node = []
        
        self.has_next = False
        node_list=[]
        
        ## generate ball candidate
        for node in ball_candidates:
            tmp = NodeCandidate(node, frame_idx)
            crop=self.crop_image(img,tmp)
            tmp.image_crop=crop
            node_list.append(tmp)
            
        if self.ball_found:  # has ball found then limit search range
            return self.track_path(node_list, frame_idx)

        else:
            self.find_lost_path(node_list, frame_idx)
            return  self.extract_path 
            # return self.traced_new_path(ball_candidates, frame_idx)
           
      
