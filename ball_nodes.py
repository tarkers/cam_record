import math
from typing import List
from xml.dom.minicompat import NodeList
from  numpy.linalg import norm
import numpy as np
import cv2
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
import collections

import numpy as np


def pooling(feature_map, size=2, stride=2):
    channel=feature_map.shape[0]
    height=feature_map.shape[1]
    width=feature_map.shape[2]
    padding_height=np.uint16(round((height-size+1)/stride))
    padding_width=np.uint16(round((width-size+1)/stride))
    print(padding_height,padding_width)

    pool_out = np.zeros((channel,padding_height,padding_width),dtype=np.uint8)
    
    for map_num in range(channel):  
        out_height = 0  
        for r in np.arange(0,height, stride):  
            out_width = 0  
            for c in np.arange(0, width, stride):  
                pool_out[map_num,out_height, out_width] = np.max(feature_map[map_num,r:r+size,c:c+size])  
                out_width=out_width+1
            out_height=out_height+1
    return pool_out

def calculate_length(p1, p2):
    vector = p2 - p1
    return math.ceil(math.sqrt(float(vector[0] ** 2 + vector[1] ** 2)))


def point_to_line(p1, p2, p3):
    return norm(np.cross(p2 - p1, p1 - p3)) / norm(p2 - p1)

def point_in_rect(point,rect):
    x1, y1, w, h = rect
    x2, y2 = x1+w, y1+h
    x, y = point
    if (x1 < x and x < x2):
        if (y1 < y and y < y2):
            return True
    return False
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

def shape_similar_test(ball,test):
    gray_ball = cv2.cvtColor(ball,cv2.COLOR_BGR2GRAY) 
    gray_image = cv2.cvtColor(test,cv2.COLOR_BGR2GRAY) 
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(gray_ball,None)
    kp2, des2 = sift.detectAndCompute(gray_image,None)
    if len(kp1)==0 or len(kp2)==0:
        return False
    
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
   
    # # Apply ratio test
    good = []
    has_match=False
    for data in matches:
        try:
            m,n =data
        except ValueError:
            return False
        # if m.distance < 0.85*n.distance:
        img1_idx = m.queryIdx
        img2_idx = m.trainIdx
    
        # x - columns
        # y - rows
        # Get the coordinates
        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt
        next_point=np.array([int(x2), int(y2)])
        if m.distance < 0.85 * n.distance:
            has_match=True
            good.append([m])


    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(ball,kp1,test,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,matchColor=None)
    cv2.namedWindow("knn",cv2.WINDOW_NORMAL)
    cv2.imshow("knn",img3)
    cv2.waitKey(0)
    
    return has_match
    
def find_ball_rect(rgb,c_x,c_y,o_x,o_y):
    candidate_rects=[]
    ## use this data to split block ##
    (nb, ng, nr) = cv2.split(rgb)
    for u in (nb, ng, nr):
        u[u>100]=u[u>100]//5+102
        u[u<100]=0
    rgb=cv2.merge([nb, ng, nr])
    #############################
    crop=cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)
    # rgb=cv2.cvtColor(crop,cv2.COLOR_GRAY2BGR)
    ## track by contour

    contours,_ = cv2.findContours(crop, 1, 2)   
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)  # 外接矩形
        bc_x,bc_y=x+w//2,y+h//2
        c_len=calculate_length(np.array([bc_x,bc_y]),np.array([c_x,c_y]))
        # if  (w>8 and h>8 and c_len<15) or point_in_rect([c_x,c_y],[x, y, w, h]):    
        if  w>8 and h>8 and  point_in_rect([c_x,c_y],[x, y, w, h]):  
            # cv2.rectangle(rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)  
            candidate_rects.append([o_x+(x-c_x),o_y+(y-c_y),w,h])
    # cv2.circle(rgb, (c_x,c_y), 5, (0,0,255), -1) 
    cv2.imshow("ball contour",rgb)
    cv2.waitKey(0)    
    return candidate_rects

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
        
   
         

    # def add_new_node(self,new_node):
    #     new_node.is_on_path=True
    #     self.extract_path[0].rank = self.extract_path[0].rank + 1
    #     self.extract_path[0].child_node = new_node
    #     new_node.parent_node = self.extract_path[0]
    #     self.extract_path.insert(0,new_node)
    #     self.path_line=[new_node.point] + self.path_line
    
    def add_new_node(self,new_node):
        new_node.is_on_path=True
        last_start=self.extract_path[-1]
        last_start.rank = last_start.rank + 1
        last_start.child_node = new_node
        new_node.parent_node = last_start
        self.extract_path.append(new_node)
        


    def crop_image(self,img,node,crop_size=None):
        if crop_size is None:
            crop_size=node.size//2
        x,y=node.point
        h,w,c=img.shape
        img=img.copy()
        crop =img[max(y-crop_size,0):min(h,y+crop_size),max(x-crop_size,0):min(w,x+crop_size)]
        return crop
    

    
    def match_keypoints(self, ball_candidates,obj_rects, frame_idx,img,mask):
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
            print("check ball found")
            self.track_path(node_list,obj_rects, frame_idx,img,mask)
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
                        
                        if start_node.rank > 6:
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
                            if self.check_ball_trajectory(path_line):
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
            self.ball_candidiates.remove(node)
            # print(f"{node.ID}->node_remove")
            if node.child_node is not None:
                child_node = node.child_node
                # self.ball_candidiates.append(child_node)  # become root node
                # print(f"{child_node.ID} is new root")
                tmp = child_node
                while tmp.child_node is not None:
                    tmp.child_node.rank -= 1
                    tmp = tmp.child_node
            del node

        if self.ball_found:
            self.ball_candidiates=[]
        
    def track_path(self,node_list,obj_rects, frame_idx,img,mask=None):
        '''
        when we found ball_candidates we start to narror serch area 
        contour rect check
        '''
        now_start=self.extract_path[-1]
        found_next=False

        ## track by ball
        for new_candidate in node_list:
            distance = calculate_length(now_start.point, new_candidate.point) 
            if distance >0 and distance < 15 * min((frame_idx-now_start.frame_id ),1):
                self.add_new_node(new_candidate)
                found_next=True
                break
        
        
        if not found_next:
            self.ball_found=False
            x,y=now_start.point
            mh,mw=mask.shape
            ball_image=now_start.image_crop
            print("check point",(x,y))
            c_size=300
            lh,uh,lw,uw=max(0,y-c_size),min(mh,y+c_size),max(0,x-c_size),min(mw,x+c_size)
            crop=img[lh:uh,lw:uw]
            c_y,c_x=min(y,c_size),min(x,c_size)
            new_candidate=find_ball_rect(crop,c_x,c_y,x,y)
            if len(new_candidate):
                print("has ttt:",len(new_candidate))
                for cnd in new_candidate:
                    x,y,w,h=cnd
                    ##test knn ##
                    crop_test=img[y:y+h,x:x+w,:]
                    if w <25 and h<25 and shape_similar_test(ball_image.copy(),crop_test.copy()): # this means it might just crop ball ,will change to fine point
                        print(w,h)
                        xys=[x+w//2,y+h//2,min(max(x//2,y//2),min(w,h))]
                        tmp = NodeCandidate(xys, frame_idx)
                        tmp.image_crop=self.crop_image(img,tmp)
                        self.add_new_node(tmp)
                    else:
                        print("is in bigger range")
                    # crop_test=img[y:y+h,x:x+w,:]
                    # self.count_histograms(ball_image,crop_test)
                    # self.count_histograms(ball_image,crop_test,c_x,c_y)
                    # cv2.imshow("t",crop_test)
                    # cv2.waitKey(0)
            # for cnd in new_candidate:
            #     print(cnd,[x,y])
            #     ## test ##
            #     x,y,w,h=cnd
            #     xys=[x+w//2,y+h//2,min(max(x//2,y//2),now_start.size)]
            #     tmp = NodeCandidate(xys, frame_idx)
            #     crop=self.crop_image(img,tmp)
            #     self.add_new_node(tmp)
                self.ball_found=True
            
        if not self.ball_found:
            print("no found path data")
        return
    

    
    def check_ball_trajectory(self,path_line):
        path_line=np.array(path_line)
        y_diff_time,x_diff_time =0,0
        
        pass_times=max(len(path_line)//3,3) 
        
        for idx in range(1,len(path_line)-1):
            d_1=np.array(path_line[idx]-path_line[idx-1])
            
            d_2=np.array(path_line[idx+1]-path_line[idx])
            
            y_diff_time+=(d_1[1]*d_2[1])<0
            x_diff_time+=(d_1[0]*d_2[0])<0
        # print("path_line:",path_line)
        # print("y:",y_diff_time)
        # print("x:",x_diff_time)
        
        return y_diff_time<pass_times and x_diff_time<pass_times
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

    def count_histograms(self,rgb,crop):
        (nb, ng, nr) = cv2.split(rgb)
        (bb, bg, br) = cv2.split(crop)
        for u in (nb, ng, nr,bb, bg, br):
            u[u>100]=u[u>100]//5+102
            u[u>20 and u<100]=40
        rgb=cv2.merge([nb, ng, nr])
        crgb=cv2.merge([bb, bg, br])
        test=cv2.cvtColor(crgb,cv2.COLOR_BGR2GRAY)
        contours,_ = cv2.findContours(test, 1, 2)   
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)  # 外接矩形
            bc_x,bc_y=x+w//2,y+h//2
            # c_len=calculate_length(np.array([bc_x,bc_y]),np.array([c_x,c_y]))
            # # if  (w>8 and h>8 and c_len<15) or point_in_rect([c_x,c_y],[x, y, w, h]):    
            if  w>8 and h>8 :  
                cv2.rectangle(crgb, (x, y), (x + w, y + h), (0, 255, 0), 2)  
                # candidate_rects.append([o_x+(x-c_x),o_y+(y-c_y),w,h])
        # cv2.circle(rgb, (c_x,c_y), 5, (0,0,255), -1) 
        cv2.namedWindow("ball",cv2.WINDOW_NORMAL)
        cv2.imshow("ball",crgb)
        cv2.waitKey(0)    
        
        # cv2.namedWindow("rgb",cv2.WINDOW_NORMAL)
        # cv2.imshow("rgb",rgb)
        # cv2.namedWindow("tcrop",cv2.WINDOW_NORMAL)
        # cv2.imshow("tcrop",crgb)
        # cv2.waitKey(0)
        
        
    def knn_match(self,ball_image,next_image,c_x,c_y):
        '''
        need to find volley ball from next image
        '''
        (nb, ng, nr) = cv2.split(next_image)
        (bb, bg, br) = cv2.split(ball_image)
        gray_next = cv2.cvtColor(next_image,cv2.COLOR_BGR2GRAY) # queryImage
        gray_ball = cv2.cvtColor(ball_image,cv2.COLOR_BGR2GRAY) # queryImage
        
        n_h = cv2.hconcat([nb, ng, nr])
        b_h = cv2.hconcat([bb, bg, br])
        cv2.namedWindow("n_h",cv2.WINDOW_NORMAL)
        cv2.imshow("n_h",n_h)
        cv2.namedWindow("b_h",cv2.WINDOW_NORMAL)
        cv2.imshow("b_h",b_h)
        cv2.waitKey(0)
        # Initiate SIFT detector
       
        return []