import json
import numpy as np
import copy

def crop_scale(motion, scale_range=[1, 1]):
    '''
        Motion: [(M), T, 17, 3].
        Normalize to [-1, 1]
    '''
    result = copy.deepcopy(motion)
    valid_coords = motion[motion[..., 2]!=0][:,:2]
    if len(valid_coords) < 4:
        return np.zeros(motion.shape)
    xmin = min(valid_coords[:,0])
    xmax = max(valid_coords[:,0])
    ymin = min(valid_coords[:,1])
    ymax = max(valid_coords[:,1])
    ratio = np.random.uniform(low=scale_range[0], high=scale_range[1], size=1)[0]
    scale = max(xmax-xmin, ymax-ymin) * ratio
    if scale==0:
        return np.zeros(motion.shape)
    xs = (xmin+xmax-scale) / 2
    ys = (ymin+ymax-scale) / 2
    result[...,:2] = (motion[..., :2]- [xs,ys]) / scale
    result[...,:2] = (result[..., :2] - 0.5) * 2
    result = np.clip(result, -1, 1)
    return result

def halpe2h36m(x):
    '''
        Input: x (T x V x C)  
       //Halpe 26 body keypoints
    {0,  "Nose"},
    {1,  "LEye"},
    {2,  "REye"},
    {3,  "LEar"},
    {4,  "REar"},
    {5,  "LShoulder"},
    {6,  "RShoulder"},
    {7,  "LElbow"},
    {8,  "RElbow"},
    {9,  "LWrist"},
    {10, "RWrist"},
    {11, "LHip"},
    {12, "RHip"},
    {13, "LKnee"},
    {14, "Rknee"},
    {15, "LAnkle"},
    {16, "RAnkle"},
    {17,  "Head"},
    {18,  "Neck"},
    {19,  "Hip"},
    {20, "LBigToe"},
    {21, "RBigToe"},
    {22, "LSmallToe"},
    {23, "RSmallToe"},
    {24, "LHeel"},
    {25, "RHeel"},
    '''
    T, V, C = x.shape
    y = np.zeros([T,17,C])
    y[:,0,:] = x[:,19,:]
    y[:,1,:] = x[:,12,:]
    y[:,2,:] = x[:,14,:]
    y[:,3,:] = x[:,16,:]
    y[:,4,:] = x[:,11,:]
    y[:,5,:] = x[:,13,:]
    y[:,6,:] = x[:,15,:]
    y[:,7,:] = (x[:,18,:] + x[:,19,:]) * 0.5
    y[:,8,:] = x[:,18,:]
    y[:,9,:] = x[:,0,:]
    y[:,10,:] = x[:,17,:]
    y[:,11,:] = x[:,5,:]
    y[:,12,:] = x[:,7,:]
    y[:,13,:] = x[:,9,:]
    y[:,14,:] = x[:,6,:]
    y[:,15,:] = x[:,8,:]
    y[:,16,:] = x[:,10,:]
    return y
    
def read_input(json_path, vid_size, scale_range, focus):
    with open(json_path, "r") as read_file:
        results = json.load(read_file)
    kpts_all = []
    for item in results:
        if focus!=None and item['idx']!=focus:
            continue
        kpts = np.array(item['keypoints']).reshape([-1,3])
        kpts_all.append(kpts)
    kpts_all = np.array(kpts_all)
    kpts_all = halpe2h36m(kpts_all)
    if vid_size:
        w, h = vid_size
        scale = min(w,h) / 2.0
        kpts_all[:,:,:2] = kpts_all[:,:,:2] - np.array([w, h]) / 2.0
        kpts_all[:,:,:2] = kpts_all[:,:,:2] / scale
        motion = kpts_all
    if scale_range:
        motion = crop_scale(kpts_all, scale_range) 
    return motion.astype(np.float32)


def rebuild_result_ids(result_path,tracklet_path):
    '''
    rebuild all result_ids from the result_path
    '''
    with open(tracklet_path, "r") as read_file:
        tracklets = json.load(read_file)
        
    with open(result_path, "r") as read_file:
        results = json.load(read_file)
    dupilicate_ids=np.array(list(tracklets.keys())).astype(int)
    for item in results:
        if item['idx'] in dupilicate_ids:
            print(item['idx'])
            item['idx']=tracklets[str(item['idx'])]
    
    with open("test_rematch.json", 'w') as json_file:
        json_file.write(json.dumps(results))    

def rebuild_tracklet_ids(tracklet_path):
    '''
    change trackletto match root
    '''
        
    with open(tracklet_path, "r") as read_file:
        tracklets = json.load(read_file)
    track=np.array(tracklets["tracklet"])
    track=np.unique(track,axis=0)
    track=np.sort(track, axis=1)
    track=np.sort(track, axis=0)
    in_group={}
    for ids in track:
        if ids[0] not in in_group:
           in_group[ids[0]]=ids[0] 
           in_group[ids[1]]=ids[0] 
        else:   #if already in group
           in_group[ids[1]]=in_group[ids[0]] 
    for key in in_group.copy():
        if float(key)==in_group[key]:
            del in_group[key]
    with open("match_ids.json", 'w') as json_file:
        json_file.write(json.dumps(in_group))      




if __name__ == '__main__':
    rebuild_result_ids(r"D:\Chen\cam_record\results.json",r"D:\Chen\cam_record\match_ids.json")