import os
import numpy as np
from Tracker import Tracking
from Sequence import Sequence_t
from PrRe import PrRe
from Iou import estimateIOU, estimateCenterErr
import logging
import multiprocessing
from multiprocessing import Pool
import warnings

warnings.filterwarnings('ignore')
class uav123_tracker(Tracking):
    def __init__(self, name: str, path='/ssd3/lz/NIPS2022/workspace/UAV20L/results', reverse=False):
        super().__init__(name, path, reverse)
        self._prsu = uav123_prsu()

    def prebox(self, sequence):
        boxtxt = os.path.join(self._path, '{}.txt'.format(sequence))
        if not os.path.exists(boxtxt):
            boxtxt = os.path.join(self._path, '{}_001.txt'.format(sequence))
        try:
            with open(boxtxt, 'r') as f:
                pre_value = np.loadtxt(f, delimiter=',', skiprows=1)
        except:
            with open(boxtxt, 'r') as f:
                pre_value = np.loadtxt(f, delimiter='\t', skiprows=1) 
        return pre_value
    


class uav123_prsu(PrRe):
    def __init__(self):
        super().__init__()
        super().reset()
        self.reset()
        self.max_center_errot = 20  # lasot 20 pixel
        self.max_overlap = 1
        self.LXaxis = np.linspace(0, self.max_center_errot, 100)
        self.Xaxis = np.linspace(0, 0.999999, 100)
    
    def reset(self):
        self.center_error = []
        self.overlaps = []

    def add_list_center_error(self, center_error:list):
        self.center_error=np.concatenate((self.center_error, center_error))

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
    
    def precision(self):
        try:
            pre = [
                np.sum(i <= thres
                       for i in self.center_error).astype(float) / self.count
                for thres in self.LXaxis
            ]
        except Exception as e:
            print(e)

        return np.trapz(np.array(pre), x=self.LXaxis)*100  / self.max_center_errot
    
    def add_list_iou(self, overlap: list):
        super().add_list_iou(overlap)


    def AO(self):
        return np.mean(self.overlaps)

class uav123_sequence(Sequence_t):
    def __init__(self, name: str, dataset='/ssd3/lz/dataset/UAV123/'):
        super().__init__(name, dataset)

    def uav123_gt(self, type="UAV123"):
        
        gtfile = os.path.join(self.dataset, "anno", type,"{}.txt".format(self.name))
        with open(gtfile, 'r') as f:
            value = np.loadtxt(f, delimiter=',')
        
        return value 


def compute_tpr_curves(trajectory: uav123_tracker, sequence: uav123_sequence, all_prsu: uav123_prsu, reverse=False):

    #overlaps = np.array(calculate_overlaps(trajectory, sequence.groundtruth(), (sequence.size) if bounded else None))
    prebbox= trajectory.prebox(sequence.name)
    gt = sequence.uav123_gt(type="UAV20L")
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

    center_error = np.concatenate(([0], np.array([estimateCenterErr(prebbox_sub[i], gt_sub[i]) for i in range(len(prebbox_sub))])))


    all_prsu.add_list_iou(overlaps)
    all_prsu.add_list_center_error(center_error)

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
   
    su = tracker._prsu.success()
    pre = tracker._prsu.precision()
    ao = tracker._prsu.AO()
    # print('Trackers: {} success: {} precesion: {} AO:{}'. format(tracker.name, su, pre)) 
    print('Trackers: {}     AO:{}'. format(tracker.name, ao)) 
##success or pr-r
seqlist = []
# with open('/ssd3/lz/dataset/UAV123/UAV20L.txt', 'r') as f:
with open('/ssd3/lz/dataset/UAV123/UAV20L.txt', 'r') as f:
    seq_value = f.readlines()
    for val in seq_value:
        seqlist.append(val.split("\n")[0])

tracker_list = os.listdir('/ssd3/lz/NIPS2022/workspace/UAV20L/results')
all_trackers = [uav123_tracker(tracker) for tracker in tracker_list]
all_sequence = [uav123_sequence(seq, dataset='/ssd3/lz/dataset/UAV123/') for seq in seqlist]

# for i, trackers in enumerate(all_trackers):
#     print(trackers.name)
#     for sequence in all_sequence:
#         # if not sequence.name == 'pot_indoor':
#             # continue
#         if sequence.name in trackers._seqlist:
#             compute_tpr_curves(trackers, sequence, trackers._prsu)
#             #print('{}: length of iou {} '.format(trackers.name, trackers._prre.count))
#         else:
#             trackers.lack(sequence.name)
#             continue
   
#     su = trackers._prsu.success()
#     pre = trackers._prsu.precision()
#     print('Trackers: {} success: {} precesion: {}'. format(trackers.name, su, pre))



# pool = Pool(processes=multiprocessing.cpu_count())    # set the processes max number 3
pool = Pool(processes=10)    # set the processes max number 3
for i in range(len(all_trackers)):
    result = pool.apply_async(analysis_tracker, (i,))
pool.close()
pool.join()
if result.successful():
    print("successful")
