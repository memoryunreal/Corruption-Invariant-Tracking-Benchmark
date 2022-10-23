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

# newdata = "/home/dataset4/cvpr2023/vot2020-C/sequences/"
# origindata_dir = "/home/dataset/vot2020/sequences/"

newdata = "/home/dataset4/cvpr2023/votlt2020-C/sequences/"
origindata_dir = "/home/dataset/votlt2020/sequences/"

seqlist = os.listdir(origindata_dir)
seqlist.remove('list.txt')
# seqlist.remove('list.txt.bk') # vot2020
seqlist.sort()
seqlist = ["group1",
"parachute",
"rollerman",
"skiing",
"volkswagen",
"yamaha",
]
originseq_color = [os.path.join(origindata_dir,seq,"color") for seq in seqlist] 
# origindata_depth = [os.path.join(origindata_dir,seq, 'depth') for seq in seqlist] 

newseq_color = [os.path.join(newdata, seq, "color") for seq in seqlist]
# newdata_depth = [os.path.join(newdata, seq, 'depth') for seq in seqlist]
# origin_num = 0
# new_num =0

originseq_color.sort()
newseq_color.sort()
# origindata_depth.sort()
# newdata_depth.sort()

def generate_corpfiles(i):
    corp_trans = CIT() # initialize the corruption type and level for one sequence
    print(corp_trans.corrupt_func.__name__, corp_trans.level)

    colorfiles = os.listdir(originseq_color[i])
    colorlist = [] # create an empty list
    
    for imgfile in colorfiles: # remove other suffix file in the color directory 
        if os.path.splitext(imgfile)[1] == '.jpg': 
            colorlist.append(imgfile)

    
    colorlist.sort() # sort the image files


    if not os.path.exists(newseq_color[i]):  # mkdir for new directory
        os.makedirs(newseq_color[i])



    if firstframe_clean: # for specified corruption type firstframe_clean or firstframe corrupted
        shutil.copy2(os.path.join(originseq_color[i],colorlist[0]), os.path.join(newseq_color[i], colorlist[0]))
        colorlist = colorlist[1:]
    elif firstframe_corrupt:
        colorlist = [colorlist[0]]

    for img_idx in range(len(colorlist)):
        
        ori_colorfile = colorlist[img_idx]
 
        imagefile = os.path.join(originseq_color[i],ori_colorfile)
        corrupted_color = corp_trans.corrupt_trans(imagefile)
        corrupted_color.save(os.path.join(newseq_color[i], ori_colorfile))

firstframe_clean = False
firstframe_corrupt = False
pool = Pool(processes=multiprocessing.cpu_count())    # set the processes max number 3
# pool = Pool(processes=1)    # set the processes max number 3
for i in range(len(seqlist)):
    result = pool.apply_async(generate_corpfiles, (i,))
pool.close()
pool.join()
if result.successful():
    print("successful")


