#coding=utf-8
#!/usr/bin/python
import vot
from vot import Rectangle
#'''
import sys
import os
prj_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(prj_path)
from os.path import realpath, dirname, join
import argparse
# del os.environ['MKL_NUM_THREADS']

import time
import cv2
import torch
import numpy as np

from tracker.TSDMTrack import TSDMTracker
from tools.bbox import get_axis_aligned_bbox

SiamRes_dir = os.path.join(prj_path, "weight/modelRes.pth") #modelMob
SiamMask_dir = os.path.join(prj_path, "weight/Res20.pth") #Mob20.pth'
Dr_dir = os.path.join(prj_path, "weight/High-Low-two.pth.tar")


parser = argparse.ArgumentParser(description='GPU selection and SRE selection', prog='tracker')
parser.add_argument("--gpu", default=0, type=int)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)



# start to track
handle = vot.VOT("rectangle",'rgbd')
region = handle.region()
image_file1, image_file2 = handle.frame()
if not image_file1:
    sys.exit(0)

image_rgb = cv2.imread(image_file1)
image_depth = cv2.imread(image_file2, -1)
if len(image_depth.shape) == 3:  
    image_depth = cv2.cvtColor(image_depth, cv2.COLOR_BGR2GRAY)


tracker = TSDMTracker(SiamRes_dir, SiamMask_dir, Dr_dir, image_rgb, image_depth, region)

#track
while True:
    image_file1, image_file2 = handle.frame()
    if not image_file1:
        break
    image_rgb = cv2.imread(image_file1)
    image_depth = cv2.imread(image_file2, -1)
    if len(image_depth.shape) == 3:  
        image_depth = cv2.cvtColor(image_depth, cv2.COLOR_BGR2GRAY)
    state = tracker.track(image_rgb, image_depth)
    region_Dr, score = state['region_Siam'], state['score'] #state['region_Dr']
    if score > 0.3:
        score = 1
    else:
        score = 0

    handle.report(Rectangle(region_Dr[0],region_Dr[1],region_Dr[2],region_Dr[3]), score)
'''
handle = vot.VOT("rectangle",'rgbd')
region = handle.region()
image_file1, image_file2 = handle.frame()
while True:
    handle.report(Rectangle(2,2,1,1), 1)
'''


