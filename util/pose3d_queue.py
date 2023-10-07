import cv2
from threading import Thread
from itertools import count
from queue import Queue
import numpy as np
import os
import pathlib
import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch.nn as nn
from tqdm import tqdmf
from matplotlib import pyplot as plt
import copy
import tqdm

## our folder##
from Pose3D.lib.utils.tools import *
from Pose3D.lib.utils.learning import load_backbone
from Pose3D.lib.utils.utils_data import flip_data
from Pose3D.dataset_wild import *

class Pose3DGenerator:
    """
    generate pose queue
    """

    def __init__(self, cfg=None):
        self.cfg = cfg
        self.results_all=[]
        self.pause_stream=False
        self.video_info=None
        self.load_model()

        
    def load_model(self):
        model_backbone = load_backbone(self.cfg)
   
        if torch.cuda.is_available():
            model_backbone = nn.DataParallel(model_backbone)
            model_backbone = model_backbone.cuda()

        print('Loading checkpoint', self.cfg.evaluate)
        checkpoint = torch.load(self.cfg.evaluate, map_location=lambda storage, loc: storage)
        model_backbone.load_state_dict(checkpoint['model_pos'], strict=True)
        self.model = model_backbone
        self.model.eval()
        self.testloader_params = {
            'batch_size': 1,
            'shuffle': False,
            'num_workers': 8,
            'pin_memory': True,
            'prefetch_factor': 4,
            'persistent_workers': True,
            'drop_last': False
        }
        
    

        
    def prepare_ids(self,folder_path,out_path):
        json_paths=glob.glob(rf"{folder_path}/*/*.json")
        for json in json_paths:
            id_txt=os.path.basename(json).split('.')[0]
            if not str.isnumeric(id_txt):
                continue
            basename =self.video_info['name']+f"_id_{id_txt}"
            results_all=self.perpare_3d_joints(json)

            self.save_3D_Pose(out_path,basename,results_all)






    def perpare_3d_joints(self,json_path,for_eval=True):   
        if not wild_dataset:
            return
        wild_dataset = PoseDataset(json_path, clip_len=self.cfg.clip_len, scale_range=[1,1], focus=self.cfg.focus)
        test_loader = DataLoader(wild_dataset, **self.testloader_params)
        results_all=[]
        with torch.no_grad():
            for batch_input in tqdm(test_loader):
                N, T = batch_input.shape[:2]
                if torch.cuda.is_available():
                    batch_input = batch_input.cuda()
                if self.cfg.no_conf:
                    batch_input = batch_input[:, :, :, :2]
                if self.cfg.flip:    
                    batch_input_flip = flip_data(batch_input)
                    predicted_3d_pos_1 = self.model(batch_input)
                    predicted_3d_pos_flip = self.model(batch_input_flip)
                    predicted_3d_pos_2 = flip_data(predicted_3d_pos_flip) # Flip back
                    predicted_3d_pos = (predicted_3d_pos_1 + predicted_3d_pos_2) / 2.0
                else:
                    predicted_3d_pos = self.model(batch_input)
                if self.cfg.rootrel:
                    predicted_3d_pos[:,:,0,:]=0                    # [N,T,17,3]
                else:
                    predicted_3d_pos[:,0,0,2]=0
                    pass
                if self.cfg.gt_2d:
                    predicted_3d_pos[...,:2] = batch_input[...,:2]
                results_all.append(predicted_3d_pos.cpu().numpy())
        results_all = np.hstack(results_all)
        results_all = np.concatenate(results_all)     

        return results_all
        
    
        
    def save_3D_Pose(self,out_path,basename,results_all,for_eval=True):
        folder=rf"{out_path}/{basename}"
        p = pathlib.Path(folder)
        p.mkdir(parents=True, exist_ok=True)
        self.render_3D_plot(results_all, '%s/%s.mp4' % (folder,basename), keep_imgs=False, fps=self.video_info['fps'])
        
        if self.cfg.pixel:
            # Convert to pixel coordinates
            results_all = results_all * (min(self.video_info['frameSize']) / 2.0)
            results_all[:,:,:2] = results_all[:,:,:2] + np.array(self.video_info['frameSize']) / 2.0
        np.save('%s/%s.npy' % (folder,basename), results_all)
    
    
    def render_3D_plot(self,motion_input, save_path, keep_imgs=False, fps=60, color="#F96706#FB8D43#FDB381", with_conf=False, draw_face=False):
        ensure_dir(os.path.dirname(save_path))
        motion = copy.deepcopy(motion_input)
        if motion.shape[-1]==2 or motion.shape[-1]==3:
            motion = np.transpose(motion, (1,2,0))   #(T,17,D) -> (17,D,T) 
        
        motion_world = self.pixel2world_vis_motion(motion, dim=3)
        
        self.motion2video_3d(motion_world, save_path=save_path, keep_imgs=keep_imgs, fps=fps)

    def motion2video_3d(self,motion, save_path, fps=60, keep_imgs = False):
    #     motion: (17,3,N)

        #create writer
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        writer = cv2.VideoWriter(save_path, fourcc, fps, (1000, 1000))

        # videowriter = imageio.get_writer(save_path, fps=fps)
        vlen = motion.shape[-1]
        # save_name = save_path.split('.')[0]
        # frames = []
        joint_pairs = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [8, 11], [8, 14], [9, 10], [11, 12], [12, 13], [14, 15], [15, 16]]
        joint_pairs_left = [[8, 11], [11, 12], [12, 13], [0, 4], [4, 5], [5, 6]]
        joint_pairs_right = [[8, 14], [14, 15], [15, 16], [0, 1], [1, 2], [2, 3]]
        
        color_mid = "#00457E"
        color_left = "#02315E"
        color_right = "#2F70AF"
        
        ## init flag
        fig = plt.figure(0, figsize=(10, 10))
        ax = plt.axes(projection="3d")
        
        plt.ion()
        
        for f in tqdm(range(vlen)):
            j3d = motion[:,:,f]
            ax.set_xlim(-512, 0)
            ax.set_ylim(-256, 256)
            ax.set_zlim(-512, 0)
            # ax.set_xlabel('X')
            # ax.set_ylabel('Y')
            # ax.set_zlabel('Z')
            ax.view_init(elev=12., azim=80)
            plt.tick_params(left = False, right = False , labelleft = False ,
                            labelbottom = False, bottom = False)
            for i in range(len(joint_pairs)):
                limb = joint_pairs[i]
                xs, ys, zs = [np.array([j3d[limb[0], j], j3d[limb[1], j]]) for j in range(3)]
                if joint_pairs[i] in joint_pairs_left:
                    ax.plot(-xs, -zs, -ys, color=color_left, lw=3, marker='o', markerfacecolor='w', markersize=3, markeredgewidth=2) # axis transformation for visualization
                elif joint_pairs[i] in joint_pairs_right:
                    ax.plot(-xs, -zs, -ys, color=color_right, lw=3, marker='o', markerfacecolor='w', markersize=3, markeredgewidth=2) # axis transformation for visualization
                else:
                    ax.plot(-xs, -zs, -ys, color=color_mid, lw=3, marker='o', markerfacecolor='w', markersize=3, markeredgewidth=2) # axis transformation for visualization
                
            #test drawing
            fig.canvas.draw()
            img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
            
            # frame_vis = get_img_from_fig(fig)
            cv2.imshow("plot",img)
            k = cv2.waitKey(33) & 0xFF
            if k == 27:
                break
            writer.write(img)
            plt.cla()
            writer.release()

