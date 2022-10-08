import random
import os
from turtle import color
import cv2
import os
from mytest.corrupt_transform import Corrupt_Transform as CIT
from PIL import Image
import shutil
from multiprocessing import Pool
import multiprocessing
from time import sleep
# newdata = "/home/dataset/GOT-10K-C/firstframe-corrupt/"
# newdata = "/home/dataset/vot2020-C/sequences/"
# newdata = "/home/dataset/votlt2020-C/sequences/"
# newdata = "/home/dataset/depthtrack-C/sequences/"
# newdata = "/home/dataset/UAV123-C/data_seq/UAV123/"
newdata = "/home/dataset4/cvpr2023/GOT-10K-C/"
# newdata = "/home/dataset/vot2020-C/sequences/"
# newdata = "/home/dataset/votlt2020-C/sequences/"
# newdata = "/home/dataset/depthtrack-C/sequences/"
origindata_dir = "/home/dataset/GOT-10K/val/"
# origindata_dir = "/home/dataset/UAV123/data_seq/UAV123/"
# origindata_dir = "/home/dataset/votlt2020/sequences/"
# origindata_dir = "/home/dataset/alldata"
# origindata_dir = "/home/dataset/vot2020/sequences/"

seqlist = os.listdir(origindata_dir)
seqlist.remove('list.txt')
# with open('/home/dataset/UAV123-C/UAV20L_list.txt') as f:
# with open('/home/dataset/depthtrack-C/sequences/list.txt') as f:
    # seqlist = []
    # values = f.readlines()
    # for val in values:
        # seqlist.append(val.split("\n")[0])
# seqlist = ['yogurt_indoor']
seqlist.sort()
# seqlist = seqlist[:71] # GOT-10K
# seqlist = [ "GOT-10k_Val_000018", ] 
# seqlist = seqlist[:31] # vot2022
# seqlist = seqlist[:31] # votlt2022
# with open(os.path.join(newdata, 'list.txt'), 'w') as f:
    # for seq in seqlist:
        # f.writelines(seq+'\n')

origindata_color = [os.path.join(origindata_dir,seq) for seq in seqlist] 
# origindata_depth = [os.path.join(origindata_dir,seq, 'depth') for seq in seqlist] 

newdata_color = [os.path.join(newdata, seq) for seq in seqlist]
# newdata_depth = [os.path.join(newdata, seq, 'depth') for seq in seqlist]
# origin_num = 0
# new_num =0

origindata_color.sort()
newdata_color.sort()
# origindata_depth.sort()
# newdata_depth.sort()


# for i in range(len(seqlist)):
#     corp_trans = CIT()
#     colorlist = os.listdir(origindata_color[i])
#     # depthlist = os.listdir(origindata_depth[i])
#     colorlist.sort()
#     colorlist.remove('groundtruth.txt')
#     if not os.path.exists(newdata_color[i]):
#         os.makedirs(newdata_color[i])
#     # depthlist.sort()


#     for img_idx in range(len(colorlist)):
        
#         ori_colorfile = colorlist[img_idx]
#         # groundtruth txt copy
#         shutil.copy2(os.path.join(origindata_color[i], 'groundtruth.txt'), os.path.join(newdata_color[i], 'groundtruth.txt'))
#         imagefile = os.path.join(origindata_color[i],ori_colorfile)
#         corrupted_color = corp_trans.corrupt_trans(imagefile)
#         corrupted_color.save(os.path.join(newdata_color[i], ori_colorfile))

def generate_corpfiles(i):
    corp_trans = CIT()
    print(corp_trans.corrupt_func.__name__, corp_trans.level)
    colorlist = os.listdir(origindata_color[i])
    # depthlist = os.listdir(origindata_depth[i]) 
    colorlist.sort()
    # depthlist.sort()
    colorlist.remove('groundtruth.txt') # GOT-10K
    colorlist.remove('absence.label') # GOT-10K
    colorlist.remove('cover.label') # GOT-10K
    colorlist.remove('cut_by_image.label') # GOT-10K
    colorlist.remove('meta_info.ini') # GOT-10K
    if not os.path.exists(newdata_color[i]): 
        os.makedirs(newdata_color[i])
    # if not os.path.exists(newdata_depth[i]): 
        # os.makedirs(newdata_depth[i])
    
    # depthlist.sort()

    if firstframe_clean:
        shutil.copy2(os.path.join(origindata_color[i],colorlist[0]), os.path.join(newdata_color[i], colorlist[0]))
        colorlist = colorlist[1:]
    elif firstframe_corrupt:
        colorlist = [colorlist[0]]
    # shutil.copy2(os.path.join(origindata_color[i], 'groundtruth.txt'), os.path.join(newdata_color[i], 'groundtruth.txt')) # GOT-10k
    # shutil.copy2(os.path.join(origindata_color[i], 'cover.label'), os.path.join(newdata_color[i], 'cover.label')) # GOT-10k
    # shutil.copy2(os.path.join(origindata_color[i], 'absence.label'), os.path.join(newdata_color[i], 'absence.label')) # GOT-10k
    # shutil.copy2(os.path.join(origindata_color[i], 'meta_info.ini'), os.path.join(newdata_color[i], 'meta_info.ini')) # GOT-10k
    # shutil.copy2(os.path.join(origindata_color[i], 'cut_by_image.label'), os.path.join(newdata_color[i], 'cut_by_image.label')) # GOT-10k
    for img_idx in range(len(colorlist)):
        
        ori_colorfile = colorlist[img_idx]
        # groundtruth txt copy
        # shutil.copy2(os.path.join(origindata_depth[i], depthlist[img_idx]), os.path.join(newdata_depth[i], depthlist[img_idx]))
        imagefile = os.path.join(origindata_color[i],ori_colorfile)
        
        corrupted_color = corp_trans.corrupt_trans(imagefile)
        corrupted_color.save(os.path.join(newdata_color[i], ori_colorfile))

firstframe_clean = False
firstframe_corrupt = True
pool = Pool(processes=multiprocessing.cpu_count())    # set the processes max number 3
# pool = Pool(processes=1)    # set the processes max number 3
for i in range(len(seqlist)):
    result = pool.apply_async(generate_corpfiles, (i,))
pool.close()
pool.join()
if result.successful():
    print("successful")



