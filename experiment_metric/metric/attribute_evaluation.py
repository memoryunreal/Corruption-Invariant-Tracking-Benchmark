import os
import numpy as np
from Tracker import Tracking
from Sequence import Sequence_t
from PrRe import PrRe
from Iou import estimateIOU
import logging
log_path = os.path.join(os.path.split(os.path.realpath(__file__))[0], '../log/attribute.log')
logging.basicConfig(filename=log_path, level=logging.DEBUG, filemode='w', format='%(levelname)s:%(asctime)s:%(message)s', datefmt='%Y-%d-%m %H:%M:%S')
def compute_ATT_curves(trajectory: Tracking, sequence: Sequence_t, all_prre: PrRe, stc_flag=False):

    #overlaps = np.array(calculate_overlaps(trajectory, sequence.groundtruth(), (sequence.size) if bounded else None))
    prebbox, confidence = trajectory.prebox_conf(sequence.name)
    gt = sequence.gt 

    # firstframe in each sequence
    overlaps = np.concatenate(([1], np.array([estimateIOU(prebbox[i], gt[i+1] ) for i in range(len(prebbox))])))
    overlaps[np.isnan(overlaps)]=0
    confidence = np.concatenate(([1], np.array(confidence)))


    #n_visible = len([region for region in sequence.groundtruth() if region.type is not RegionType.SPECIAL])
    # sequence.invisible (full-occlusion tag) if invisible= 1 full-occlusion invisible
    visible = np.array(sequence.invisible) < 1
    visible = visible + 0
    try:
        assert len(overlaps) == len(visible) == len(confidence)
    except:
        print("assert not equal")
    all_attribute = attribute_index(sequence._path, len(overlaps), stc_flag)

    all_prre.add_attribute(all_attribute)    
    all_prre.add_list_iou(overlaps)
    all_prre.add_visible(visible)
    all_prre.add_confidence(confidence) 

def attribute_index(sequencepath, frame_num, stc_flag):
    
    '''
        return: dict {tag} [value]
    '''
    detattribute_list = ['aspect-change.tag', 'background-clutter.tag','depth-change.tag','fast-motion.tag', 'dark-scene.tag', 'moving-view.tag',
                    'deformable.tag', 'out-of-plane.tag', 'partial-occlusion.tag', 'reflective-target.tag', 'size-change.tag', 'similar-objects.tag', 'unassigned.tag']
    cdtbattribute_list = ['fast-motion.tag', 'size-change.tag', 'aspect-change.tag', 'partial-occlusion.tag', 'similar-objects.tag', 'out-of-plane.tag', 'depth-change.tag', 'reflective-target.tag',
                    'deformable.tag', 'dark-scene.tag', 'unassigned.tag']
    stcattribute_list = ['IV','CDV','DV','DDV','SV','SDC','SCC','BCC','BSC','SD','PO']
    allattribute_list = list(set(detattribute_list + cdtbattribute_list))
    value_list = []
    att_name = []
    # allattribute_list = ['backg.tag']
    # for i, attribute in enumerate(allattribute_list):
    for i, attribute in enumerate(cdtbattribute_list):

        # if attribute == 'out-of-frame.tag':
        #     print("ok")
        attpath = os.path.join(sequencepath, attribute)
        try:
            with open(attpath, 'r') as f:
                value = np.loadtxt(f)
            assert len(value) == frame_num
        except:
            value = np.zeros(frame_num)
        value_list.append(value)
        att_name.append(cdtbattribute_list[i])
    ### stc benchmark
    if stc_flag:
        stcattpath = os.path.join(sequencepath, 'Overall_annotation.txt')
        try:
            with open(stcattpath, 'r') as f:
                stcvalue = np.loadtxt(f,delimiter=',')
                stcattr_value = [[] for i in range(len(stcattribute_list))]
                for index, stcval in enumerate(stcvalue):
                    for id, val in enumerate(stcval):
                        stcattr_value[id].append(val)
                # add stc attribute to all attribute
                for index, val in enumerate(stcattr_value):
                    value_list.append(val)
                    att_name.append(stcattribute_list[index])
        except:
            stcvalue = [np.zeros(frame_num) for i in range(len(stcattribute_list))]
            for index, val in enumerate(stcvalue):
                value_list.append(val)
                att_name.append(stcattribute_list[index])
    
    dict_attribute = dict(zip(att_name, value_list))
    return dict_attribute

### success precsion or precision-recall
stc_flag = True
# seq_list = os.listdir('/data1/yjy/rgbd_benchmark/alldata')
seq_list = os.listdir('/ssd3/lz/MM2022/dataset/stc')
seq_list.remove('list.txt')
# all_trackers = [Tracking(tracker) for tracker in os.listdir('/data1/yjy/rgbd_benchmark/all_benchmark/results/')]
# all_trackers = [Tracking(tracker) for tracker in ["DAL", "TSDM", "DeT","CMCT"]]
# all_trackers = [Tracking(tracker) for tracker in ["CMCT","DAL","DeT","keeptrack_origin_default","dimp_origin","atom_origin","TSDM","CA3DMS","DSKCF","DSKCF_shape","prdimp_origin","CSRKCF","transt50"]]
# all_trackers = [Tracking(tracker) for tracker in ["CMCT","DAL","DeT","OTR","dimp_origin","atom_origin","TSDM","CA3DMS","DSKCF","DSKCF_shape","CSRKCF"]]
all_trackers = [Tracking(tracker) for tracker in ["dimp_origin","atom_origin","prdimp_origin","keeptrack_origin_default"]]
all_sequence = [Sequence_t(seq) for seq in seq_list]

output_tracker = []
output_fscore = [[] for i in range(26)]
tag_list = []
for trackers in all_trackers:
    print(trackers.name)
    for sequence in all_sequence:
        
        if sequence.name in trackers._seqlist:
            compute_ATT_curves(trackers, sequence, trackers._prre, stc_flag)
            #print('{}: length of iou {} '.format(trackers.name, trackers._prre.count))
        else:
            trackers.lack(sequence.name)
            continue
    if stc_flag:
        result = trackers._prre.su_AT
    else:
        result = trackers._prre.fscore_AT
    print('Trackers: {}'.format(trackers.name))
    for index, tag in enumerate(result):
        print('{}   fscore: {}'.format(tag, result.get(tag)))
        output_fscore[index].append(result.get(tag))
        logging.info('Trackers: {} attribute:{}   fscore: {}'.format(trackers.name, tag, result.get(tag)))
        tag_list.append(tag)
    output_tracker.append(trackers.name)
print('Trackers: ', output_tracker)
outputfile = open('/ssd3/lz/MM2022/ACM-MM2022/stc_attribute.txt', 'w')
outputfile.writelines('Trackers: {}'.format(output_tracker))
outputfile.writelines('\n')
for i in range(len(output_fscore)):
    str_output = '{} fscore: {}'.format(tag_list[i], output_fscore[i])
    outputfile.writelines(str_output)
    outputfile.writelines('\n')

outputfile.close()