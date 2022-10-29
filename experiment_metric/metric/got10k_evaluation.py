import os
import argparse
from matplotlib.pyplot import box
import numpy as np
from Tracker import Tracking
from Sequence import Sequence_t
from PrRe import PrRe
from Iou import estimateIOU, estimateCenterErr
import logging
import multiprocessing
from multiprocessing import Pool
import warnings
import sys
import cv2

sys.path.append("/home/lz/anaconda3/envs/nbconda/lib/python3.9/site-packages/")
from vot.dataset import Dataset, DatasetException, Sequence, BaseSequence, PatternFileListChannel
from vot.region.io import write_trajectory, read_trajectory, parse_region

from vot.utilities import Progress, localize_path, read_properties, write_properties

def make_full_size(x, output_sz):
    """
    zero-pad input x (right and down) to match output_sz
    x: numpy array e.g., binary mask
    output_sz: size of the output [width, height]
    """
    if x.shape[0] == output_sz[1] and x.shape[1] == output_sz[0]:
        return x
    pad_x = output_sz[0] - x.shape[1]
    if pad_x < 0:
        x = x[:, :x.shape[1] + pad_x]
        # padding has to be set to zero, otherwise pad function fails
        pad_x = 0
    pad_y = output_sz[1] - x.shape[0]
    if pad_y < 0:
        x = x[:x.shape[0] + pad_y, :]
        # padding has to be set to zero, otherwise pad function fails
        pad_y = 0
    return np.pad(x, ((0, pad_y), (0, pad_x)), 'constant', constant_values=0)
def rect_from_mask(mask):
    """
    create an axis-aligned rectangle from a given binary mask
    mask in created as a minimal rectangle containing all non-zero pixels
    """
    x_ = np.sum(mask, axis=0)
    y_ = np.sum(mask, axis=1)
    x0 = np.min(np.nonzero(x_))
    x1 = np.max(np.nonzero(x_))
    y0 = np.min(np.nonzero(y_))
    y1 = np.max(np.nonzero(y_))
    return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]
warnings.filterwarnings('ignore')
class got10k_tracker(Tracking):
    def __init__(self, name: str, path='/ssd3/lz/NIPS2022/workspace/GOT10K/val_results', reverse=False, type=None):
        super().__init__(name, path, reverse)
        self._prsu = got10k_prsu()
        
        self.type = type
        votdir = {
            "vot2020": "unsupervised",
            "votlt": "longterm",
            "votrgbd": "rgbd-unsupervised"
        }
  
        if self.type:
            self._path = os.path.join(self._path, votdir[self.type])
    '''
        vot: unsupervised
        votlt: longterm
        votrgbd: rgbd-unsupervised
    '''
    def prebox(self, sequence):

        boxtxt = os.path.join(self._path, '{}.txt'.format(sequence))
        if not os.path.exists(boxtxt):
            boxtxt = os.path.join(self._path, '{}_001.txt'.format(sequence))
        if self.type:
            boxtxt = os.path.join(self._path, sequence,'{}.txt'.format(sequence))
            if not os.path.exists(boxtxt):
                boxtxt = os.path.join(self._path, sequence, '{}_001.txt'.format(sequence))
        if self.type == "vot":  
            try:
                with open(boxtxt, 'r') as f:
                    pre_value = np.loadtxt(f, delimiter=',', skiprows=1)
            except:

                value=[]
                # with open(gtfile, 'r') as f:
                #     for line in f.readlines():
                #         value.append(parse_region(line.strip()))
                colorimg = os.path.join("/ssd3/lz/dataset/vot2020/sequences", sequence, 'color')
                img = os.listdir(colorimg)
                image = cv2.imread(os.path.join(colorimg, img[0]))
                groundtruth = read_trajectory(boxtxt)
                for id in range(len(groundtruth)):
                    if id ==0:
                        continue
                    try:
                        mask = make_full_size(groundtruth[id].mask,(image.shape[1], image.shape[0]) )
                        rect_noxy = rect_from_mask(mask)
                        value.append([groundtruth[id]._offset[0], groundtruth[id]._offset[1], rect_noxy[2], rect_noxy[3]])
                    except:
                        value.append([np.nan, np.nan, np.nan, np.nan])
                pre_value=np.array(value)
    
        else:
            try:
                with open(boxtxt, 'r') as f:
                    pre_value = np.loadtxt(f, delimiter=',', skiprows=1)
            except:
                with open(boxtxt, 'r') as f:
                    pre_value = np.loadtxt(f, delimiter='\t', skiprows=1) 
        return pre_value


    @property
    def _seqlist(self):
        if self.type:
            return os.listdir(self._path)
        else:
            return super()._seqlist 

class got10k_prsu(PrRe):
    def __init__(self):
        super().__init__()
        super().reset()
        self.reset()
        self.max_overlap = 1
        self.Xaxis = np.linspace(0, self.max_overlap, 100)
    
    def reset(self):
        self.center_error = []
        self.overlaps = []


    def success_50(self):
        return np.sum(i > 0.5 for i in self.overlaps).astype(float) / self.count
    def success_70(self):
        return np.sum(i > 0.75 for i in self.overlaps).astype(float) / self.count
    def AO(self):
        return np.mean(self.overlaps)

    def success(self):
        try:
            succ = [
                np.sum(i >= thres
                       for i in self.overlaps).astype(float) / self.count
                for thres in self.Xaxis
            ]
        except Exception as e:
            print(e)

        return np.trapz(np.array(succ), x=self.Xaxis) *100  / self.max_overlap
     
    def add_list_iou(self, overlap: list):
        super().add_list_iou(overlap)

class got10k_sequence(Sequence_t):
    def __init__(self, name: str, dataset='/ssd3/lz/dataset/GOT-10K/', data_type=None):
        super().__init__(name, dataset)
        self.type=data_type

    def got10k_gt(self, type="val"):
        if  self.type == "vot2020" or self.type=="votlt" or self.type=="votrgbd":
            gtfile = os.path.join(self.dataset, self.name, "groundtruth.txt")
            colorimg = os.path.join(self.dataset, self.name, 'color')
            img = os.listdir(colorimg)
            image = cv2.imread(os.path.join(colorimg, img[0]))
            try:
                with open(gtfile, 'r') as f:
                    value = np.loadtxt(f, delimiter=',')
            except:
                value=[]
                # with open(gtfile, 'r') as f:
                #     for line in f.readlines():
                #         value.append(parse_region(line.strip()))
                groundtruth = read_trajectory(gtfile)
                for id in range(len(groundtruth)):
                    try:
                        mask = make_full_size(groundtruth[id].mask,(image.shape[1], image.shape[0]) )
                        rect_noxy = rect_from_mask(mask)
                        value.append([groundtruth[id]._offset[0], groundtruth[id]._offset[1], rect_noxy[2], rect_noxy[3]])
                    except:
                        value.append([np.nan, np.nan, np.nan, np.nan])
                value=np.array(value)
                
        elif self.type=="val":
            gtfile = os.path.join(self.dataset, type, self.name, "groundtruth.txt")
            with open(gtfile, 'r') as f:
                value = np.loadtxt(f, delimiter=',')
        
        return value 


def compute_tpr_curves(trajectory: got10k_tracker, sequence: got10k_sequence, all_prsu: got10k_prsu, reverse=False):

    #overlaps = np.array(calculate_overlaps(trajectory, sequence.groundtruth(), (sequence.size) if bounded else None))
    prebbox= trajectory.prebox(sequence.name)
    gt = sequence.got10k_gt()
    gt = gt[1:]
    # remove nan in groundtruth
    lost = np.isnan(gt[:,0])
    # get the subset of valid groundtruth

    subset = ~lost
    
    try:
        assert len(gt) == len(prebbox)
    except:
        print("assert gt not equal prebbox", sequence.name)  
    prebbox_sub = prebbox[subset]
    gt_sub = gt[subset]
    
    overlaps = np.concatenate(([1], np.array([estimateIOU(prebbox_sub[i], gt_sub[i] ) for i in range(len(prebbox_sub))])))
    overlaps[np.isnan(overlaps)]=0

    all_prsu.add_list_iou(overlaps)

def analysis_tracker(idx):
    tracker =  all_trackers[idx]
    # print(tracker.name)
    for sequence in all_sequence:
        # if not sequence.name == 'pot_indoor':
            # continue
        if sequence.name in tracker._seqlist:
            compute_tpr_curves(tracker, sequence, tracker._prsu)
            #print('{}: length of iou {} '.format(trackers.name, trackers._prre.count))
        else:
            tracker.lack(sequence.name)
            continue
   
    su = tracker._prsu.success_50()
    ao = tracker._prsu.AO()
    # print('Trackers: {:<30} success50: {} \t AO: {}'. format(tracker.name, su, ao)) 
    print('"Trackers: {:<30} success50: {:0.1f} \t AO: {:0.1f}",'. format(tracker.name, su*100, ao*100)) 
##success or pr-r
# with open('/ssd3/lz/dataset/UAV123/UAV20L.txt', 'r') as f:


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate the got-10k results, output SR(50)/AO")
    parser.add_argument('--dataset', type=str, help="dataset_type vot got10k votlt")
    parser.add_argument('--resultpath', type=str, help="tracker results dir")
    parser.add_argument('--allcorp', type=bool, default=False, help="tracker results dir")
    args = parser.parse_args()

    seqlist = []
    dataset = args.dataset
    result_dir = args.resultpath 
    got10k_dataset='/home/dataset4/cvpr2023/GOT-10K/'
    vot2020_dataset = '/home/dataset4/cvpr2023/vot2020/sequences/'

    all_corp_type = args.allcorp
    alltype = ["gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
        "glass_blur", "motion_blur", "zoom_blur", "contrast",
        "elastic_transform", "pixelate", "jpeg_compression", "speckle_noise",
        "gaussian_blur", "spatter", "saturate", "bit_error", "h265_crf", "h265_abr", "fog", "rain", "frost", "snow","brightness" ]

    if dataset == "got10k":
        # get got10k sequence 
        with open(os.path.join(got10k_dataset,'val/list.txt'), 'r') as f:
            seq_value = f.readlines()
            for val in seq_value:
                seqlist.append(val.split("\n")[0])
        all_sequence = [got10k_sequence(seq, dataset=got10k_dataset,data_type="val") for seq in seqlist] #got10k
        # resdir = '/ssd3/lz/NIPS2022/111.5/NIPS2022_workspace/GOT10K-C/firstframe-clean'
        
        tracker_list = os.listdir(result_dir) # 
        tracker_list.sort()
        all_trackers = [got10k_tracker(name=tracker, path=result_dir) for tracker in tracker_list]

    elif dataset == "vot2020":
        with open(os.path.join(vot2020_dataset, "list.txt"), 'r') as f:
            seq_value = f.readlines()
            for val in seq_value:
                seqlist.append(val.split("\n")[0])
        # seqlist = ["fernandsso"]
        all_sequence = [got10k_sequence(seq, dataset=vot2020_dataset, data_type="vot2020") for seq in seqlist] #got10k
        tracker_list = os.listdir(result_dir)
        tracker_list.sort()
        all_trackers = [got10k_tracker(name=tracker, path=result_dir, type="vot2020") for tracker in tracker_list]

    elif dataset == "votlt":
        with open('/ssd3/lz/dataset/votlt2020-C/sequences/list.txt', 'r') as f:
            seq_value = f.readlines()
            for val in seq_value:
                seqlist.append(val.split("\n")[0])
        seqlist.sort() 
        # seqlist = seqlist[:31] # vot2020
        all_sequence = [got10k_sequence(seq, dataset='/ssd3/lz/dataset/votlt2020/sequences/', data_type="votlt") for seq in seqlist] #got10k
        # tracker_list = os.listdir('/ssd3/lz/NIPS2022/111.5/votlt2020-C/results/') # votlt2020
        tracker_list=['mixformer']
        tracker_list.sort()
        all_trackers = [got10k_tracker(name=tracker, path="/ssd3/lz/NIPS2022/111.5/votlt2020-C/results/", type="votlt") for tracker in tracker_list]


    if all_corp_type:
        for cortype in alltype: 
            print(cortype)

            tracker_list = os.listdir(os.path.join('/ssd3/lz/NIPS2022/workspace/GOT10K/all_corruption', cortype))
            tracker_list.sort()
            tracker_result_path = os.path.join('/ssd3/lz/NIPS2022/workspace/GOT10K/all_corruption', cortype)
            all_trackers = [got10k_tracker(tracker,path=tracker_result_path) for tracker in tracker_list]

            pool = Pool(processes=1)    # set the processes max number 3
            for i in range(len(all_trackers)):
                result = pool.apply_async(analysis_tracker, (i,))
            pool.close()
            pool.join()

    else:
        # tracker_list = os.listdir('/ssd3/lz/NIPS2022/workspace/GOT10K/val_results')

        pool = Pool(processes=10)    # set the processes max number 3
        for i in range(len(all_trackers)):
            result = pool.apply_async(analysis_tracker, (i,))
        pool.close()
        pool.join()




