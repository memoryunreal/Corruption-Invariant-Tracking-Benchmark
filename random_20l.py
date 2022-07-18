import random
import os
import cv2
import os
from mytest.corrupt_transform import Corrupt_Transform as CIT
from PIL import Image
import shutil
from multiprocessing import Pool
import multiprocessing
from time import sleep
# newdata = "/home/dataset/GOT-10K-C"
# newdata = "/home/dataset/vot2020-C/sequences/"
newdata = "/home/dataset/UAV123-C/data_seq/UAV123-random/"
# newdata = "/home/dataset/UAV123-C/data_seq/UAV123/"
# origindata_dir = "/home/dataset/GOT-10K/test/"
# origindata_dir = "/home/dataset/UAV123/data_seq/UAV123/"
origindata_dir = "/home/dataset/UAV123/data_seq/UAV123/"
# origindata_dir = "/home/dataset/vot2020/sequences/"

# seqlist = os.listdir(origindata_dir)
# seqlist.remove('list.txt')
with open('/home/dataset/UAV123-C/UAV20L_list.txt') as f:
    seqlist = []
    values = f.readlines()
    for val in values:
        seqlist.append(val.split("\n")[0])
# seqlist = ['yogurt_indoor']
seqlist.sort()
# seqlist = seqlist[:71] # GOT-10K
# seqlist = seqlist[:31] # vot2022
#seqlist = seqlist[:20] # votlt2022
# with open(os.path.join(newdata, 'list.txt'), 'w') as f:
    # for seq in seqlist:
        # f.writelines(seq+'\n')
origindata_color = [os.path.join(origindata_dir,seq) for seq in seqlist] 

newdata_color = [os.path.join(newdata, seq) for seq in seqlist]
# origin_num = 0
# new_num =0
# for i in  range(len(seqlist)):
#     origin_num+= len(os.listdir(origindata_color[i]))
#     new_num+= len(os.listdir(newdata_color[i]))
# print(origin_num, new_num)
origindata_color.sort()
newdata_color.sort()

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
    # colorlist.remove('groundtruth.txt') # GOT-10K
    if not os.path.exists(newdata_color[i]): 
        os.makedirs(newdata_color[i])
    # depthlist.sort()

    for img_idx in range(200):

        ori_colorfile = colorlist[img_idx]
        # groundtruth txt copy
        # shutil.copy2(os.path.join(origindata_color[i], 'groundtruth.txt'), os.path.join(newdata_color[i], 'groundtruth.txt')) # GOT-10k
        imagefile = os.path.join(origindata_color[i],ori_colorfile)
        corrupted_color = corp_trans.corrupt_trans(imagefile)
        corrupted_color.save(os.path.join(newdata_color[i], ori_colorfile))
    
    corp_trans_1 = CIT()
    for img_idx in range(200,500):

        ori_colorfile = colorlist[img_idx]        # groundtruth txt copy
        # shutil.copy2(os.path.join(origindata_color[i], 'groundtruth.txt'), os.path.join(newdata_color[i], 'groundtruth.txt')) # GOT-10k
        imagefile = os.path.join(origindata_color[i],ori_colorfile)
        corrupted_color = corp_trans_1.corrupt_trans(imagefile)
        corrupted_color.save(os.path.join(newdata_color[i], ori_colorfile))
        
    corp_trans_2 = CIT()
    for img_idx in range(500,1000):

        ori_colorfile = colorlist[img_idx]
        # groundtruth txt copy
        # shutil.copy2(os.path.join(origindata_color[i], 'groundtruth.txt'), os.path.join(newdata_color[i], 'groundtruth.txt')) # GOT-10k
        imagefile = os.path.join(origindata_color[i],ori_colorfile)
        corrupted_color = corp_trans_2.corrupt_trans(imagefile)
        corrupted_color.save(os.path.join(newdata_color[i], ori_colorfile))
    
    corp_trans_3 = CIT()
    print("corp_3 ", corp_trans_3.corrupt_func.__name__, corp_trans_3.level)
    for img_idx in range(1000,len(colorlist)):

        ori_colorfile = colorlist[img_idx]
        # groundtruth txt copy
        # shutil.copy2(os.path.join(origindata_color[i], 'groundtruth.txt'), os.path.join(newdata_color[i], 'groundtruth.txt')) # GOT-10k
        imagefile = os.path.join(origindata_color[i],ori_colorfile)
        corrupted_color = corp_trans_3.corrupt_trans(imagefile)
        corrupted_color.save(os.path.join(newdata_color[i], ori_colorfile))


pool = Pool(processes=multiprocessing.cpu_count())    # set the processes max number 3
# pool = Pool(processes=1)    # set the processes max number 3
for i in range(len(seqlist)):
    result = pool.apply_async(generate_corpfiles, (i,))
pool.close()
pool.join()
if result.successful():
    print("successful")

