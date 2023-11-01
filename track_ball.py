import math
from typing import List
from  numpy.linalg import norm
import numpy as np
import cv2
import array
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
        self.lost_chance=2
        
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
                
                if (n_f - s_f) > 5:  # # remove lost track data 
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
                            test_idx = 0
                            
                            while tnp.parent_node is not None:
                                tnp.is_in_path=True
                                tmp=tnp.image_crop
                                self.extract_path.append(tnp)
                                self.path_line.append(tnp.point)
                                tnp = tnp.parent_node
                                cv2.imwrite(f"{test_idx}_t.jpg",tmp)
                                test_idx+=1
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

        
        test_crop=self.test[max(y-50,0):min(1050,y+50),max(x-50,0):min(1850,x+50)] 
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
            self.new_node_candidate.append(self.extract_path[0])
            print("lost ball tracking")

        ## track by image
        elif False==True:
            pass
        else:   # we need to retrace the path
            self.ball_found=False
            self.new_node_candidate.append(self.extract_path[0])    # append node back to candidate
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
    
    def match_keypoints(self, ball_candidates, frame_idx,img,test):
        self.img=img
        self.test=test
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
            self.track_path(node_list, frame_idx,img)
            return self.extract_path 

        else:
            self.find_lost_path(node_list, frame_idx)
            return  self.extract_path 
            # return self.traced_new_path(ball_candidates, frame_idx)
            
    
    # def knn_match(self,ball_image,next_image,candidate_rects):
    #     '''
    #     need to find volley ball from next image
    #     '''

    #     gray_ball = cv2.cvtColor(ball_image,cv2.COLOR_BGR2GRAY) # queryImage
    #     for rect in candidate_rects:
    #         x,y,w,h =rect
    #         next_crop=next_image[y:y+h+5,x:x+w+5,:]
    #         gray_image = cv2.cvtColor(next_crop,cv2.COLOR_BGR2GRAY) # trainImage
        
        
    #         ##next image center
    #         c_x,c_y=gray_image.shape[1]//2 ,gray_image.shape[0]//2   
    #         ##find ball contour

        
    #         min_len=25
    #         cv2.namedWindow("tests",cv2.WINDOW_NORMAL)
    #         cv2.imshow("tests",next_image)
            
    #         # Initiate SIFT detector
    #         sift = cv2.SIFT_create()
    #         # find the keypoints and descriptors with SIFT
    #         kp1, des1 = sift.detectAndCompute(gray_ball,None)
    #         kp2, des2 = sift.detectAndCompute(gray_image,None)
    #         # BFMatcher with default params
    #         bf = cv2.BFMatcher()
    #         matches = bf.knnMatch(des1,des2,k=2)
    #         ball_match =[]
    #         # # Apply ratio test
    #         good = []
    #         for m,n in matches:
    #             # if m.distance < 0.85*n.distance:
                
    #             img1_idx = m.queryIdx
    #             img2_idx = m.trainIdx
            
    #             # x - columns
    #             # y - rows
    #             # Get the coordinates
    #             (x1, y1) = kp1[img1_idx].pt
    #             (x2, y2) = kp2[img2_idx].pt
    #             next_point=np.array([int(x2), int(y2)])
    #             cal_len= calculate_length(next_point,np.array([c_x,c_y])) 
               
    #             if cal_len <min_len:
    #                 min_len=cal_len
    #                 ball_match=[[int(x+x2), int(y+y2)]]
    #                 good=[[m]]

    #         # cv2.drawMatchesKnn expects list of lists as matches.
    #     img3 = cv2.drawMatchesKnn(gray_ball,kp1,gray_image,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,matchColor=None)
    #     cv2.namedWindow("test",cv2.WINDOW_NORMAL)
    #     cv2.imshow("test",img3)
    #     cv2.waitKey(0)
    #     if len(ball_match)==0:
    #         return []
    #     print("candidate_count:",len(ball_match))
    #     return ball_match
           
      
