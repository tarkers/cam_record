import math
from  numpy.linalg import norm
import numpy as np
import cv2
def calculate_length(p1,p2):
    return round(math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2))

def point_to_line(p1,p2,p3):
    return norm(np.cross(p2-p1, p1-p3))/norm(p2-p1)

class TrackBall(object):
    def __init__(self,fps=60):
        self.ball_candidate=[]
        self.track_range={}
        self.extract_path=[]
        self.speed_minthreshold=0
        self.speed_maxthreshold=10
        self.node_candidate=[]  #x,y,size,frame_idx
        self.node_dict={}
        self.found_center=[]
        self.ball_found=False
        
    def traced_lost(self,key_point_xys,frame_idx):  # if ball not found then traced ball
        
        s_length=len(self.node_candidate)
        for start_idx in range(s_length):
            start_node=self.node_candidate[start_idx]
            s_str=str(start_node)
            
            for next_node in key_point_xys:
                length=calculate_length(start_node,next_node)  
                if length>0 and length < 20 *(frame_idx-start_node[3]):
                    if s_str not in self.track_range:   # has'nt append to path yet
                        self.track_range[s_str]=[[length,frame_idx-start_node[3]]]
                    else:       #on the path
                        if self.track_range[s_str][-1][-1] < frame_idx-start_node[3] and \
                            self.track_range[s_str][-1][0]< length:   #path found
                                self.ball_found=True
                                self.found_center=next_node+[frame_idx]
                                ###################### has path found##########################################
                                tmp_start=self.node_candidate[start_idx]
                                self.node_candidate=[tmp_start,next_node+[frame_idx]]
                                self.extract_path=[tmp_start,next_node+[frame_idx]]
                                print("data_found")
                                return self.node_candidate
                        print(rf"{[s_str,str(next_node)]},last_frame:{self.track_range[s_str]},now:{[length,frame_idx-start_node[3]]}")
                    self.node_dict[s_str][1]+=1  # add confidence for this node
                    x,y,size=next_node
                    
                elif length==0:
                    # print("node have to delete")
                    self.delete_node.append(start_node)
                else:
                    pass
                # else:
                #     node_str=str(next_node+[frame_idx])
                #     if node_str not in self.node_dict:
                #         self.node_candidate.append(next_node+[frame_idx])   # make this as ball candidate
                #         self.node_dict[str(next_node+[frame_idx])]=[frame_idx,0]
                #     elif self.node_dict[node_str]<3:
                #          self.node_dict[node_str]-=1

        #remove ball node
        for node in self.delete_node:
            self.node_candidate.remove(node)
            del self.node_dict[str(node)]  
        return self.node_candidate                  
    
    def match_keypoints(self,key_point_xys,frame_idx,test_img):
        if len(self.node_candidate)==0:     # first input set candidate inside
            for node in key_point_xys:
                self.node_dict[str(node+[frame_idx])]=[frame_idx,0]
                self.node_candidate.append(node+[frame_idx])
            return
        self.delete_node=[]
        
        self.has_next=False
        
        if self.ball_found: # has ball found then limit search range
            for node in key_point_xys:
                length=calculate_length(self.found_center[:2],node) 
                if length>0 and length < 40 *(frame_idx-self.found_center[3]):
                    self.has_next=True
                    self.extract_path.append(node+[frame_idx])
                    self.found_center=node+[frame_idx]

            if not self.has_next:
                print(f"lost track :{self.extract_path[-1][-1]} to {frame_idx} frame")
            return  self.extract_path 
        
        else:
            return self.traced_lost(key_point_xys,frame_idx)
           
        # if self.node_dict[s_str][1]<=0 and frame_idx-self.node_dict[s_str][0]>2: # delete this node from candidate
        #     self.node_candidate.remove(start_node)
        #     del self.node_dict[s_str]
        #     print(self.node_candidate)
                
 