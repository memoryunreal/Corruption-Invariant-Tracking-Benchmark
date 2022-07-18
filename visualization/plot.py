import os
import numpy as np
import sys
import cv2
'''
    gtbox: groundtruth list

    trackerlist: tracker name

'''

def plot(trackerlist, gtbox):

    trackerslist = trackerlist
    gt_bbox = gtbox 

    colorpath = os.path.join(sequence._path, 'color')

    colorimageindex = os.listdir(colorpath)
    colorimageindex.sort()
    colorimagepath = [os.path.join(colorpath, filename) for i, filename in enumerate(colorimageindex)]

    # line_color
    red = (0, 0, 255)
    green = (0, 255, 0)
    grey = (128, 128, 128)
    yellow = (10, 215, 255)
    blue = (255, 0, 0)
    puple = (128, 0, 128)
    orange = (0, 69, 255)
    line_color = [red, grey, yellow, blue, puple, orange, green]

    save_path = os.path.join(savepath, sequence.name, 'color')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i in range(1,len(gt_bbox)):
        originimage = cv2.imread(colorimagepath[i])
        # gt bbox plot
        try:
            cv2.line(originimage, (int(gt_bbox[i][0]),int(gt_bbox[i][1])), (int(gt_bbox[i][0]+gt_bbox[i][2]), int(gt_bbox[i][1])), line_color[-1])
            cv2.line(originimage, (int(gt_bbox[i][0]),int(gt_bbox[i][1])), (int(gt_bbox[i][0]), int(gt_bbox[i][1]+gt_bbox[i][3])), line_color[-1])
            cv2.line(originimage, (int(gt_bbox[i][0] + gt_bbox[i][2]),int(gt_bbox[i][1]+gt_bbox[i][3])), (int(gt_bbox[i][0]), int(gt_bbox[i][1]+gt_bbox[i][3])), line_color[-1])
            cv2.line(originimage, (int(gt_bbox[i][0] + gt_bbox[i][2]),int(gt_bbox[i][1]+gt_bbox[i][3])), (int(gt_bbox[i][0]+gt_bbox[i][2]), int(gt_bbox[i][1])), line_color[-1])
        except:
            savename = savename.split(".")[0]+'_gtloss' + '.' + savename.split(".")[1]
            

        for j in range(len(trackerslist)):
            try:
                left_x = trackers_bbox[j][i-1][0]
                left_y = trackers_bbox[j][i-1][1]
                width =  trackers_bbox[j][i-1][2]
                height = trackers_bbox[j][i-1][3]
            except Exception as e:
                print(e)

            try:
                cv2.line(originimage, (int(left_x),int(left_y)), (int(left_x+width), int(left_y)), line_color[j])
                cv2.line(originimage, (int(left_x),int(left_y)), (int(left_x), int(left_y+height)), line_color[j])
                cv2.line(originimage, (int(left_x+width),int(left_y+height)), (int(left_x+width), int(left_y)), line_color[j])
                cv2.line(originimage, (int(left_x+width),int(left_y+height)), (int(left_x), int(left_y+height)), line_color[j])
            except:
                savename = savename.split(".")[0]+'_loss_{}'.format(trackerslist[j]) + '.' +savename.split(".")[1]
                continue    
        
        cv2.imwrite(savename, originimage)


savepath= '/ssd3/lz/TMM2022/visualization/longterm/'
seq_list = os.listdir('/data1/yjy/rgbd_benchmark/alldata')
seq_list.remove('list.txt')
# all_trackers = [Tracking(tracker) for tracker in os.listdir('/data1/yjy/rgbd_benchmark/all_benchmark/results/')]
trackers_list = ['iiau_rgbd','TALGD','DAL','DeT','CA3DMS','CSRKCF']
# trackers_list = ['iiau_rgbd']
all_trackers = [Tracking(tracker) for tracker in trackers_list]
# all_trackers = [Tracking(tracker) for tracker in ['CADMS']]
all_sequence = [Sequence_t(seq) for seq in seq_list]
count_seq = 0

all_ok = ['developmentboard_indoor', '1 developmentboard_indoor',
'colacan03_indoor',
'ball20_indoor',
'glass01_indoor',
'trashcan_room_occ_1',
'box_humans_room_occ_1',
'duck03_wild',
'stick_indoor',
'humans_corridor_occ_2_B',
'box_room_occ_2',
'human_entry_occ_2',
'ball10_wild',
'humans_corridor_occ_2_A',
'box_room_occ_1',
'bag02_indoor'
]
for sequence in all_sequence:
    skip_flag = 0
    if np.sum(sequence.invisible==1) == 0:
        continue

    print('{} {} start'.format(count_seq,sequence.name))
    if sequence.name in all_ok:
        continue
    gt_box, beinvis_list, afinvis_list = divide(sequence)
    trackers_box = []
    for i, trackers in enumerate(all_trackers):
        
        if sequence.name in trackers._seqlist:
            prebox, state = trackingresult(trackers, sequence)
            trackers_box.append(prebox)
            if not state:
                skip_flag = 1
                break
            #print('{}: length of iou {} '.format(trackers.name, trackers._prre.count))
        else:
            skip_flag = 1
            print('{} lack {}'.format(trackers.name, sequence.name))
            break
    if skip_flag == 1:
        continue
    plot(sequence,trackers_box,trackers_list,gt_box,beinvis_list, afinvis_list)
    count_seq += 1
    print('{} {} finished'.format(count_seq,sequence.name))