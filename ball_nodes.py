import math
from typing import List
from xml.dom.minicompat import NodeList
from  numpy.linalg import norm
import numpy as np
import cv2
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

def calculate_length(p1, p2):
    vector = p2 - p1
    return math.ceil(math.sqrt(float(vector[0] ** 2 + vector[1] ** 2)))


def point_to_line(p1, p2, p3):
    return norm(np.cross(p2 - p1, p1 - p3)) / norm(p2 - p1)

def quadratic_coeff(p1,p2,p3):
    '''
    return the nearest quadratic eauactuib 
    y=ax^2+bx +c 
    '''
    x_1,y_1 = p1[0],p1[1]
    x_2,y_2 = p2[0],p2[1]
    x_3,y_3 = p3[0],p3[1]
    a = y_1/((x_1-x_2)*(x_1-x_3)) + y_2/((x_2-x_1)*(x_2-x_3)) + y_3/((x_3-x_1)*(x_3-x_2))

    b = (-y_1*(x_2+x_3)/((x_1-x_2)*(x_1-x_3))
         -y_2*(x_1+x_3)/((x_2-x_1)*(x_2-x_3))
         -y_3*(x_1+x_2)/((x_3-x_1)*(x_3-x_2)))

    c = (y_1*x_2*x_3/((x_1-x_2)*(x_1-x_3))
        +y_2*x_1*x_3/((x_2-x_1)*(x_2-x_3))
        +y_3*x_1*x_2/((x_3-x_1)*(x_3-x_2)))

    return a,b,c 

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
        
        self.ball_chance=2
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
        self.fps=fps
        self.speed_minthreshold = 0
        self.speed_maxthreshold = 10
        self.node_candidate = []  # x,y,size,frame_idx
        self.ball_candidiates = []
        self.path_candidate = PathCandidate()
        self.node_dict = {}
        self.found_center = []
        self.ball_found = False
        self.out_off_bound = False
        self.fake_candidate = []
        self.path_line=[]
        self.lost_chance=2
        
   
         

    def add_new_node(self,new_node):
        new_node.is_on_path=True
        self.extract_path[0].rank = self.extract_path[0].rank + 1
        self.extract_path[0].child_node = new_node
        new_node.parent_node = self.extract_path[0]
        self.extract_path.insert(0,new_node)
        self.path_line=[new_node.point] + self.path_line
        
    def track_path(self,node_list, frame_idx,img):
        '''
        when we found ball_candidates we start to narror serch area
        '''
        now_start=self.extract_path[0]
        found_next=False

        ## track by ball
        for new_candidate in node_list:
            distance = calculate_length(now_start.point, new_candidate.point) 
            
            if distance >0 and distance < 25 * min((frame_idx-now_start.frame_id ),1):
                self.add_new_node(new_candidate)
                found_next=True
                break
        
        
        if not  found_next:
            self.ball_found=False
        return
        
        
        self.ball_found=False
        # if self.lost_chance > 0 :
        #     return
        
        
        ##check contour of ball
        x,y=self.extract_path[0].point

        
        test_crop=self.mask[max(y-50,0):min(1050,y+50),max(x-50,0):min(1850,x+50)] 
        img_crop=img[max(y-50,0):min(1050,y+50),max(x-50,0):min(1850,x+50)] 
        c_x,c_y=test_crop.shape[1]//2 ,test_crop.shape[0]//2   
        candidate_rects=self.find_ball_rect(test_crop,c_x,c_y)
        print("candidate_rects:",len(candidate_rects))
        ball_match=self.knn_match(self.extract_path[1].image_crop,img_crop,candidate_rects)
        if len(ball_match):
            print("knn found")
            new_xy=ball_match[0]
            # ## node back to orignal size ##
            new_xy[0]+=(x-c_x)
            new_xy[1]+=(y-c_y)
            new_xy.append(16)
            # new_xy.append(self.extract_path[0].size)
            print(new_xy)
            tmp = NodeCandidate(new_xy, frame_idx)
            tmp.image_crop=self.crop_image(img,self.extract_path[0])
            cv2.imshow("previous",tmp.image_crop)
            cv2.waitKey(0)
            self.add_new_node(tmp)
            self.ball_found=True
            return
        return
        print("no data")
        # crop=img[max(y-100,0):min(1050,y+100),max(x-100,0):min(1850,x+100)]
        # c_x,c_y=crop.shape[1]//2 ,crop.shape[0]//2   
        # node=self.find_ball_rect(crop,c_x,c_y)
        # ## node back to orignal size ##
        # node[0]+=(x-c_x)
        # node[1]+=(y-c_y)
        # if self.ball_found:
        #     tmp=NodeCandidate(node,frame_idx)
        #     self.add_new_node(tmp)
        #     print("contour have found ball")
        #     return
        
        
        
        self.lost_chance-=1
       
        if  False==True:
            self.lost_chance =2
            self.ball_candidiates.append(self.extract_path[0])
            print("lost ball tracking")

        ## track by image
        elif False==True:
            pass
        else:   # we need to retrace the path
            self.ball_found=False
            self.ball_candidiates.append(self.extract_path[0])    # append node back to candidate
        return self.extract_path                
    
    def find_ball_rect(self,crop,c_x,c_y):
        rgb=cv2.cvtColor(crop,cv2.COLOR_GRAY2BGR)
        candidate_rects=[]
        ## track by contour
        contours,hierarchy = cv2.findContours(crop, 1, 2)   
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)  # 外接矩形
            bc_x,bc_y=x+w//2,y+h//2
            length=calculate_length(np.array([bc_x,bc_y]),np.array([c_x,c_y]))
            if w >8 and h>8 and  length<25:
            # if w >8 and h>8 and  length<20:
                print("find contour", (x, y), (x + w, y + h))        
                candidate_rects.append([x,y,w,h])
                cv2.rectangle(rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
        cv2.imshow("ball contour",rgb)
        cv2.waitKey(0)    
        return candidate_rects
    
    
    def match_pic(self):
        pass

    def crop_image(self,img,node,crop_size=None):
        if crop_size is None:
            crop_size=node.size//2
        x,y=node.point
        h,w,c=img.shape
        img=img.copy()
        crop =img[max(y-crop_size,0):min(h,y+crop_size),max(x-crop_size,0):min(w,x+crop_size)]
        return crop
    
    def filter_candidates(self):
        pass
    
    def match_keypoints(self, ball_candidates, frame_idx,img,mask):
        self.img=img.copy()
        self.mask=mask
        self.delete_node = []
        self.has_next = False
        node_list=[]
        first_detect= len(self.ball_candidiates) == 0
        
        ## generate ball candidate
        for node in ball_candidates:
            ## check ##    
            tmp = NodeCandidate(node, frame_idx)
            crop=self.crop_image(img,tmp)
            tmp.image_crop=crop
            if first_detect:
                self.ball_candidiates.append(tmp)
            else:
                node_list.append(tmp)
        
        if first_detect :   #first detection ,no set of ball
            return self.extract_path

       
        
       
            
        if self.ball_found:  # has ball found then limit search range
            self.track_path(node_list, frame_idx,img)
            return self.extract_path 

        else:
            self.find_lost_path(node_list, frame_idx)
            return  self.extract_path 
            # return self.traced_new_path(ball_candidates, frame_idx)
            
    
    def find_lost_path(self, node_list:List[NodeCandidate] , n_frame):
        '''
        check if ball candidate can be select
        '''
        can_lens = len(self.ball_candidiates)
        n_len=len(node_list)
        
        print("ball candidate now: ",can_lens)
        self.add_new_can=[]
        self.del_old_can=[]
        
        for idx in range(can_lens):
            start_node = self.ball_candidiates[idx]
            s_pt=start_node.point
            s_frame=start_node.frame_id
            has_child=False
            
            if (n_frame- s_frame)>4:    #node too old ,we don't need it anymore
                self.del_old_can.append(start_node)
                continue
            
            if self.ball_found:
                break
            
            for n_idx in range(n_len):  ## check if find a real ball path
                node=node_list[n_idx]
                n_pt=node.point     
                dist = calculate_length(s_pt, n_pt) 
                
                if dist > 40 * (n_frame- s_frame) or dist < 2:
                    continue
                else:   # child in path
                    has_child=True
                    print(s_pt,n_pt,dist)   
                    if start_node.child_node is not None:
                        sc_frame=start_node.child_node.frame_id
                        if sc_frame == n_frame: # means need to check who is real
                            print("need to check who is real child")     
                    elif node.is_on_path: # means there is another tree node
                        print("need to see who is real parent")
                        x,y=n_pt
                        node_xys=[x,y,node.size]
                        tmp = NodeCandidate(node_xys, n_frame)
                        tmp.image_crop=node.image_crop
                    else:   ## maybe should check knn or others later 
                       
                        node.rank=start_node.rank+1
                        node.parent_node=start_node
                        start_node.child_node=node
                        node.is_on_path=True
                        
                        if start_node.rank > 5:
                            print(f"found ball path:{start_node.ID} {start_node.point}")
                            path_line=[]
                            path_candidate=[]
                            ### test path ###
                            tnp = node
                            while tnp.parent_node is not None:
                                print(tnp.point,tnp.rank)
                                tmp=tnp.image_crop
                                path_candidate.append(tnp)
                                path_line.append(tnp.point)
                                tnp = tnp.parent_node
                            if self.check_ball_trajetory(path_line):
                                self.ball_found = True
                                while len(path_candidate):
                                    tmp=path_candidate.pop()  
                                    self.extract_path.append(tmp)
                            else:   #not correct path
                                while len(path_candidate):
                                    tmp=path_candidate.pop()  
                                    if tmp.rank<2 :
                                        tmp.ball_chance -=1
                                    else:
                                        tmp.parent_node = None
                                        tmp.child_node=None
                                        tmp.rank=0
                                self.ball_found=False
                                
                            # else:
                            #     self.ball_found=False
                            #     self.extract_path=self.extract_path[:end]
                        
                  
                
                       
            if has_child:
                self.del_old_can.append(start_node)
                    
            else:   # remove this ball candidate
                start_node.ball_chance -=1
                if start_node.ball_chance<=0:
                    self.del_old_can.append(start_node)
                    
        for node in node_list:  ## append new candidate
            if node not in self.ball_candidate:
                self.ball_candidiates.append(node)    
                               
        ## remove old candidates ##
        for node in self.del_old_can:
            print("CEHCEK")
            self.ball_candidiates.remove(node)
            print(f"{node.ID}->node_remove")
            if node.child_node is not None:
                child_node = node.child_node
                # self.ball_candidiates.append(child_node)  # become root node
                print(f"{child_node.ID} is new root")
                tmp = child_node
                while tmp.child_node is not None:
                    tmp.child_node.rank -= 1
                    tmp = tmp.child_node
            del node
            
    def check_ball_trajetory(self,path_line):
        path_line=np.array(path_line)
        y_diff_time,x_diff_time =0,0
        
        pass_times=max(len(path_line)//3,3) 
        
        for idx in range(1,len(path_line)-1):
            d_1=np.array(path_line[idx]-path_line[idx-1])
            
            d_2=np.array(path_line[idx+1]-path_line[idx])
            
            y_diff_time+=(d_1[1]*d_2[1])<0
            x_diff_time+=(d_1[0]*d_2[0])<0
        print("path_line:",path_line)
        print("y:",y_diff_time)
        print("x:",x_diff_time)
        
        
        # s = 1.0 # smoothness parameter
        # k = 2 # spline order
        # nest = -1 # estimate of number of knots needed (-1 = maximal)
        # t, u = splprep([path_line[:,0], path_line[:,1]], s=s, k=k, nest=-1)
        # # a,b,c=quadratic_coeff(path_line[-1], path_line[-2],path_line[-4])
        # # x_test=1485
        # # print(a * x_test**2 + b * x_test+ c)
        # xn, yn = splev(np.linspace(0, 1, 30), t)
        
        # # xn,yn=path_line[:,0],path_line[:,1]
        # # plt.plot(xn, yn, color='r', linewidth=1)
        # peak_x, _ = find_peaks(xn, height=50)
        # low_x, _ = find_peaks(-xn)
        # plt.plot(xn)
        # plt.plot(peak_x,xn[peak_x],"x")
        # plt.plot(low_x,xn[low_x],"x")
        
        # peaks, _ = find_peaks(yn, height=50)
        # peaks2, _ = find_peaks(-yn)
        # plt.plot(yn)
        # plt.plot(peaks,yn[peaks],"o")
        # plt.plot(peaks2,yn[peaks2],"o")
        # plt.show()
        # time.sleep(2)
        return y_diff_time<pass_times and x_diff_time<pass_times