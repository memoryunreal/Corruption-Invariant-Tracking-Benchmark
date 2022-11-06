import random
import os
from turtle import color
import cv2
from mytest.corrupt_transform import Corrupt_Transform as CIT
from PIL import Image
import shutil
from multiprocessing import Pool
import multiprocessing
from time import sleep
import numpy as np
import sys
prj_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
tool_dir = os.path.abspath(os.path.join(prj_dir, "tool_utils"))
sys.path.append(tool_dir)
from track_mix.augmix import augmix
# newdata = "/home/dataset/GOT-10K-C/firstframe-corrupt/"
augdata1 = "/home/dataset4/cvpr2023/got10k_trackmix/aug1/train"
augdata2 = "/home/dataset4/cvpr2023/got10k_trackmix/aug2/train"

origindata_dir = "/home/dataset/GOT-10K/train/"

seqlist = os.listdir(origindata_dir)
seqlist.remove('list.txt')

seqlist.sort()
seqlist = ["GOT-10k_Train_003961"]
originseq_color = [os.path.join(origindata_dir,seq) for seq in seqlist] 

aug1_color = [os.path.join(augdata1, seq) for seq in seqlist]
aug2_color = [os.path.join(augdata2, seq) for seq in seqlist]

originseq_color.sort()
aug1_color.sort()
aug2_color.sort()

class augmix_transform():
    def __init__(self, level=0, width=3, depth=-1, alpha=1.):
        super().__init__()
        self.level = level
        self.width = width
        self.depth = depth
        self.alpha = alpha

    def transform_image(self, image):
        img = augmix(np.asarray(image) / 255)
        img = np.clip(img * 255., 0, 255).astype(np.uint8)
        img = Image.fromarray(img)
        return img


def generate_augment(i):
    aug_1 = augmix_transform()
    aug_2 = augmix_transform()

    colorfiles = os.listdir(originseq_color[i])
    colorlist = [] # create an empty list
    
    for imgfile in colorfiles: # remove other suffix file in the color directory 
        if os.path.splitext(imgfile)[1] == '.jpg': 
            colorlist.append(imgfile)

    
    colorlist.sort() # sort the image files


    if not os.path.exists(aug1_color[i]):  # mkdir for new directory
        os.makedirs(aug1_color[i])
    if not os.path.exists(aug2_color[i]):  # mkdir for new directory
        os.makedirs(aug2_color[i])


    for img_idx in range(len(colorlist)):
        
        ori_colorfile = colorlist[img_idx]

        imagefile = os.path.join(originseq_color[i],ori_colorfile)
        img = Image.open(imagefile)

        # save aug file
        aug1color = aug_1.transform_image(img)
        aug1color.save(os.path.join(aug1_color[i], ori_colorfile))

        aug2color = aug_2.transform_image(img)
        aug2color.save(os.path.join(aug2_color[i], ori_colorfile))
    
    print("{} finished!!!!!!".format(imagefile))
    print("{} finished!!!!!!".format(imagefile), file=completefile)


completefile = open("/home/CVPR2023/Corruption-Invariant-Tracking-Benchmark/dataset-generate/got-10k_train/complete.log", 'a')
# pool = Pool(processes=multiprocessing.cpu_count())    # set the processes max number 3
pool = Pool(processes=1)    # set the processes max number 3
# pool = Pool(processes=multiprocessing.cpu_count())   # set the processes max number 3
for i in range(len(seqlist)):
    result = pool.apply_async(generate_augment, (i,))
pool.close()
pool.join()
if result.successful():
    print("successful")

print("all completed!")
completefile.close()

