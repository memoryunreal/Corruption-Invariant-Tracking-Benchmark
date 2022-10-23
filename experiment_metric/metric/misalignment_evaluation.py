import os
import numpy as np
from Tracker import Tracking
from Sequence import Sequence_t
from PrRe import PrRe
from Iou import estimateIOU
import logging
log_path = os.path.join(os.path.split(os.path.realpath(__file__))[0], '../log/misalignment.log')
logging.basicConfig(filename=log_path, level=logging.DEBUG, filemode='w', format='%(levelname)s:%(asctime)s:%(message)s', datefmt='%Y-%d-%m %H:%M:%S')
def compute_noise_curves(trajectory: Tracking, sequence: Sequence_t, all_prre: PrRe, tre=1):
    
    #overlaps = np.array(calculate_overlaps(trajectory, sequence.groundtruth(), (sequence.size) if bounded else None))
    prebbox, confidence = trajectory.vot_prebox_conf(sequence.name)
    gt = sequence.gt 

    # firstframe in each sequence
    try:
        overlaps = np.concatenate(([1], np.array([estimateIOU(prebbox[i], gt[i+tre] ) for i in range(len(prebbox))])))
        overlaps[np.isnan(overlaps)]=0
        confidence = np.concatenate(([1], np.array(confidence)))
    except:
        print("Error tre: {}  sequence {} sequence frame {} result frame {}".format(tre, sequence.name, len(sequence.gt), len(prebbox)))
        return 0

    #n_visible = len([region for region in sequence.groundtruth() if region.type is not RegionType.SPECIAL])
    # sequence.invisible (full-occlusion tag) if invisible= 1 full-occlusion invisible
    visible = np.array(sequence.invisible) < 1
    visible = visible + 0
    '''
        temporal evaluation
    '''

    visible = visible[tre-1:]
    '''
        temporal evaluation
    '''
    try:
        assert len(overlaps) == len(visible) == len(confidence)
    except:
        print("assert not equal")
        print("Error tre: {} Tracker: {} sequence {} sequence frame {} result frame {}".format(tre,trajectory.name, sequence.name, len(sequence.gt), len(prebbox)))
        return False    

    all_prre.add_list_iou(overlaps)
    all_prre.add_visible(visible)
    all_prre.add_confidence(confidence) 


listfile = '/ssd3/lz/TMM2022/dataset/alldata/list.txt'
seq_list = []
with open(listfile, 'r') as f:
    value = f.readlines()
    for val in value:
        seq_list.append(val.split('\n')[0])
'''
    edit three variables
    noise_workspace: vot worksapce path
    noise_list:      different workspace name
    trackers:        tracker name
'''
asynchronize_workspace = '/ssd3/lz/TMM2022/workspace/misalignment/'
asynchronize_list = ['htrans10','htrans-10','vtrans10','vtrans-10']
# asynchronize_list = ['htrans10','vtrans-10']
# trackers = ['CSRRGBD++','DSKCF_shape', 'DAL', 'DeT', 'TSDM', 'DRefine']
trackers = [ 'DAL', 'DeT', 'DRefine','TSDM', 'CSRRGBD++','DSKCF_shape']

syn_average_fscore = [[] for i in range(len(trackers))]
syn_average_pr = [[] for i in range(len(trackers))]
syn_average_re = [[] for i in range(len(trackers))]
for syn in asynchronize_list:
        all_trackers = [Tracking(tracker, path=os.path.join(asynchronize_workspace, syn, 'results')) for i,tracker in enumerate(trackers)]

        all_sequence = [Sequence_t(seq) for seq in seq_list]
        
        for id, tracker in enumerate(all_trackers):
            
            print(tracker.name)
            tracker_seq_number = 0
            for sequence in all_sequence:
          
                if not os.path.exists(os.path.join(tracker._votpath, sequence.name, '{}_001.txt'.format(sequence.name))):
                    continue
                if sequence.name in tracker._votseqlist:
                    compute_noise_curves(tracker, sequence, tracker._prre)
                    tracker_seq_number += 1
                    #print('{}: length of iou {} '.format(trackers.name, trackers._prre.count))
                else:
                    tracker.lack(sequence.name)
                    continue
                
            pr,re,fscore = tracker._prre.fscore
            if tracker.name == trackers[id]:
                syn_average_fscore[id].append(fscore)
                syn_average_pr[id].append(pr)
                syn_average_re[id].append(re)
            print('Trackers: {} trans: {} seq: {} pr: {} re: {} fscore: {}'.format(tracker.name, syn, tracker_seq_number, pr, re, fscore))


for i, sre in enumerate(syn_average_fscore):
    trackerID = trackers[i]
    srefscore = np.mean(np.array(sre))
    srepr = np.mean(np.array(syn_average_pr[i]))
    srere = np.mean(np.array(syn_average_re[i]))
    print('Trackers: {} trans pr:{} re: {} fscore: {}'.format(trackerID, srepr, srere, srefscore))

    logging.info('Trackers: {} trans pr:{} re: {} fscore: {}'.format(trackerID, srepr, srere, srefscore))



print('ok')