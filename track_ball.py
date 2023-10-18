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

    def __init__(self, node:list, frame_id:int, parent_node:'NodeCandidate'=None , ChildNode:'NodeCandidate'=None):
        global id_track
        '''
        node:[x,y,size]
        '''
        self.parent_node = parent_node
        self.child_node = ChildNode
        
        self._point = np.array(node[:2])
        self.frame_id = frame_id
        self.size = node[2]
        self.rank = 0
       
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

    def __init__(self):
        self.node_list = []
        
    def add_node(self, node):
        self.node_list.add(node)
        
    def remove_node(self, node):
        self.node_list.remove(node)
    
    
class TrackBall(object):

    def __init__(self, fps=60):
        self.ball_candidate = []
        self.track_range = {}
        self.extract_path = []
        self.speed_minthreshold = 0
        self.speed_maxthreshold = 10
        self.node_candidate = []  # x,y,size,frame_idx
        self.new_node_candidate = []
        self.node_dict = {}
        self.found_center = []
        self.ball_found = False
        self.out_off_bound = False
        self.fake_candidate = []
    
    def filter_node(self):
        print("need filter node")
    
    def find_lost_path(self, ball_candidates:List[NodeCandidate]  , n_f):
        '''
        check if ball candidate can be select
        '''
        s_length = len(self.new_node_candidate)
        node_list = []
        # ## generate ball candidate
        for tmp in ball_candidates:
            node_list.append(NodeCandidate(tmp, n_f))
        
        for start_idx in range(s_length):
            start_node = self.new_node_candidate[start_idx]
            s_f = start_node.frame_id
            
            if start_node.child_node is not None:  # means it's on path
                continue
            
            for next_node in node_list:  # check if can generate tracklet
                distance = calculate_length(start_node.point, next_node.point) 
                if distance < 1:  # not track of static object
                    if start_node not in self.delete_node:
                        self.delete_node.append(start_node)
                    continue
                if distance > 20 * (n_f - s_f) // 2:  # distance too far , start node may not be real track
                    if next_node not in self.new_node_candidate:
                        self.new_node_candidate.append(next_node)  # next_node may be root ball node
                    continue
                
                if (n_f - s_f) > 3:  # # remove lost track data 
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
                            ### test path###
                            tnp = next_node
                            self.extract_path = []
                            while tnp.parent_node is not None:
                                print(tnp.point)
                                self.extract_path.append(tnp.point)
                                tnp = tnp.parent_node
                            print("-----------")
                            break
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
        
        print(f"========================== {len(self.new_node_candidate)}")    
        return self.node_candidate         

    def track_path(self):
        pass
    
    def traced_new_path(self, ball_candidates, frame_idx):  # if ball not found then traced ball
        '''
        check if ball candidate can be select
        '''
        s_length = len(self.node_candidate)
        
        for start_idx in range(s_length):
            start_node = self.node_candidate[start_idx]
            s_str = str(start_node)
            
            for next_node in ball_candidates:  # check if can generate tracklet
                distance = calculate_length(start_node, next_node) 
                if distance == 0:  # not track of static object
                    self.delete_node.append(start_node)
                    continue
                if distance > 20 * (frame_idx - start_node[3]):  # distance too far ,may not be real track
                    # # check if candidate has no chance to be save 
                    continue
                
                if distance > 0 and distance < 10 * (frame_idx - start_node[3]):
                    if s_str not in self.track_range: 
                        self.track_range[s_str] = [[distance, frame_idx - start_node[3]]]
                    else:  # on the path
                        if self.track_range[s_str][-1][-1] < frame_idx - start_node[3] and \
                            self.track_range[s_str][-1][0] < distance:  # path found
                                self.ball_found = True
                                self.found_center = next_node + [frame_idx]
                                ###################### has path found##########################################
                                tmp_start = self.node_candidate[start_idx]
                                self.node_candidate = [tmp_start, next_node + [frame_idx]]
                                # self.extract_path = [tmp_start, next_node + [frame_idx]]
                                print("data_found")
                                return self.node_candidate
                        print(rf"{[s_str,str(next_node)]},last_frame:{self.track_range[s_str]},now:{[distance,frame_idx-start_node[3]]}")
                    self.node_dict[s_str][1] += 1  # add confidence for this node
                    x, y, size = next_node

                # else:
                #     node_str=str(next_node+[frame_idx])
                #     if node_str not in self.node_dict:
                #         self.node_candidate.append(next_node+[frame_idx])   # make this as ball candidate
                #         self.node_dict[str(next_node+[frame_idx])]=[frame_idx,0]
                #     elif self.node_dict[node_str]<3:
                #          self.node_dict[node_str]-=1

        # remove ball node
        for node in self.delete_node:
            self.node_candidate.remove(node)
            del self.node_dict[str(node)]  
        return self.node_candidate                  
    
    def match_pic(self):
        pass

    def match_keypoints(self, ball_candidates, frame_idx):
        if len(self.node_candidate) == 0:  # first input set candidate inside
            for node in ball_candidates:
                self.node_dict[str(node + [frame_idx])] = [frame_idx, 0]
                self.node_candidate.append(node + [frame_idx])
                tmp = NodeCandidate(node, frame_idx)
                self.new_node_candidate.append(tmp)
            return self.node_candidate
        self.delete_node = []
        
        self.has_next = False
        
        if self.ball_found:  # has ball found then limit search range
            self.track_path()
            # for node in ball_candidates:
            #     distance = calculate_length(self.found_center[:2], node) 
            #     if distance > 0 and distance < 40 * (frame_idx - self.found_center[3]):
            #         self.has_next = True
            #         self.extract_path.append(node + [frame_idx])
            #         self.found_center = node + [frame_idx]

            # if not self.has_next:
            #     print(f"lost track :{self.extract_path[-1][-1]} to {frame_idx} frame")
            # return  self.extract_path 
            return  self.extract_path 
        else:
            self.find_lost_path(ball_candidates, frame_idx)
            return  self.extract_path 
            # return self.traced_new_path(ball_candidates, frame_idx)
           
        # if self.node_dict[s_str][1]<=0 and frame_idx-self.node_dict[s_str][0]>2: # delete this node from candidate
        #     self.node_candidate.remove(start_node)
        #     del self.node_dict[s_str]
        #     print(self.node_candidate)
 
